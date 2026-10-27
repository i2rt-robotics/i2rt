// CAN frames over an EtherCAT-to-CAN board, driven by SOEM.
//
// The board is an EtherCAT slave with two CAN channels. Each cycle the master writes one 86-byte
// image to it and reads one back, in the same layout both ways:
//
//   byte 0      frame count (unused, 0)
//   byte 1      extended-id flag for the whole image (0 = standard 11-bit ids)
//   bytes 2..85 six 14-byte frame slots: u32 id (LE), u8 rtr, u8 dlc, u8 data[8]
//               slots 0-2 go out on CAN channel 1, slots 3-5 on channel 2; dlc 0 = empty slot
//
// So at most three frames per channel per cycle. Measured on the board (traces on a 1 kHz cycle):
//
//   - a reply arrives two cycles after its request; one whose id matches the request's lands in that
//     request's slot, one that does not (a register reply to a 0x7FF request) in whichever slot of the
//     channel the board picks;
//   - an input slot keeps its last reply for ~25 cycles when nothing new arrives, then empties;
//   - several frames with the same id on one channel in one cycle get no reply at all;
//   - input slot k accepts only frames whose id matches the last frame output slot k sent, and keeps
//     that filter while the slot stays empty. A reply matching no slot's filter is dropped. A command
//     primes its own reply's filter; a reply with another id (register access, whose requests go to
//     0x7FF) needs a slot primed with that id first -- which raw mode's per-frame `slot` and remote
//     frames (`rtr`) make possible.
//
// A stale slot cannot be told from a fresh one by its content, so in cyclic mode a reply is believed
// only if slot k carried a frame two cycles earlier, and only from the motor that frame went to. That rejects a stale reply only if some other
// motor refreshed the slot in between, so a channel must carry at least three motors and every
// slot must change hands every cycle: a rotating group does, and a group of exactly three is
// shifted one slot per cycle. With fewer than three, a dead motor's own last reply could sit in its
// slot and match again, so such a channel is refused.
//
// ponytail: if every motor on a channel stops at once (motor power cut, board CAN side hung) the
// last replies stay in the slots and some are re-read as fresh. No command reaches those motors
// then either; the motors' own CAN TIMEOUT register is the failsafe for that case.
//
// This file moves bytes and nothing else: it knows no motor protocol. Two modes share one real-time loop:
//
//   cyclic  n motors, each with a latched 8-byte frame that is re-sent every time its turn comes
//           round (three per channel per cycle, rotating), and a latest-reply buffer per motor,
//           routed by (slave, channel, reply id). A watchdog swaps in each motor's release frame
//           when the host stops refreshing the latched ones.
//   raw     no motors: frames are queued and sent once each (one per id per channel per cycle), and
//           every received frame is queued for the host once -- a slot re-read unchanged is the
//           same frame. For register access and other request/response tools.
//
// `loopback` replaces the EtherCAT exchange with an emulation of the board as measured: each output
// frame comes back as its own reply two cycles later, in the same slot, and a slot with nothing to
// answer keeps its last content. Frames to a muted id get no reply, standing in for a dead motor. The
// loop, rotation, routing, stale-slot rejection and watchdog are all testable with no hardware.

extern "C" {
#include "ethercat.h"
}

#include <arpa/inet.h>
#include <net/if.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kImageBytes = 86;
constexpr int kSlots = 6;
constexpr int kSlotBytes = 14;
constexpr int kSlotsPerChannel = 3;
constexpr int kReplyLag = 2;  // cycles from a frame going out to its reply being in the input image
constexpr int kMaxSlaves = 5;
constexpr int kRxTimeoutUs = 500;  // per cycle; SOEM's own EC_TIMEOUTRET (2 ms) is longer than a cycle
constexpr size_t kRawQueueCap = 4096;
constexpr uint16_t kEtherTypeEcat = 0x88A4;

// SOEM's ec_ API keeps its master state in globals, so one bus per process.
// ponytail: the ecx_ context API is the upgrade path for two boards on two NICs in one process.
std::atomic<bool> g_bus_open{false};

int64_t mono_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

double realtime_s() {
    // Reply stamps are wall-clock: the Python side compares them against time.time().
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return double(ts.tv_sec) + ts.tv_nsec * 1e-9;
}

void set_err(char* err, int errlen, const char* fmt, ...) {
    if (err == nullptr || errlen <= 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, size_t(errlen), fmt, ap);
    va_end(ap);
}

struct RawFrame {
    uint8_t slave, channel, dlc, rtr;
    int8_t slot;  // raw send only: the channel slot (0-2) to go out on, or -1 for the first free one
    uint32_t id;
    uint8_t data[8];
    double stamp;
};

struct Bus {
    bool loopback = false;
    bool raw = false;
    int n = 0;  // cyclic motors
    int used_slaves = 0;
    int period_us = 1000;
    int watchdog_ms = 100;
    int prio = 99;
    int cpu = -1;
    int expected_wkc = 0;

    std::vector<int> m_slave, m_channel;
    std::vector<uint32_t> m_id;
    std::vector<std::vector<int>> groups;  // [slave * 2 + channel - 1] -> motor indices
    std::vector<int> ring;

    uint8_t iomap[4096];
    uint8_t lb_out[kMaxSlaves][kImageBytes];  // loopback images
    uint8_t lb_in[kMaxSlaves][kImageBytes];
    uint8_t lb_hist[kReplyLag][kMaxSlaves][kImageBytes];  // outputs the emulated board has yet to answer
    std::vector<uint32_t> lb_muted;                        // loopback ids that never answer

    // Id each output slot carried on this cycle and the kReplyLag before it; -1 = empty. A reply in
    // input slot k is only believed from the id sent[kReplyLag][s][k] was sent to.
    int64_t sent[kReplyLag + 1][kMaxSlaves][kSlots];
    uint64_t cycle = 0;
    uint8_t raw_seen[kMaxSlaves][kSlots][kSlotBytes];  // raw mode: last content read from each slot

    // Diagnostics: the output and input image of board 0 for the next trace_want cycles.
    std::vector<uint8_t> trace;
    size_t trace_want = 0;

    std::mutex mu;  // guards tx, release, rx*, raw queues, last_set_ns
    std::vector<uint8_t> tx, release, rx;
    std::vector<double> rx_time;
    std::vector<uint64_t> rx_count;
    int64_t last_set_ns = 0;
    bool tripped = false;
    std::deque<RawFrame> raw_tx, raw_rx;
    std::condition_variable raw_cv;

    std::atomic<uint64_t> ticks{0}, overruns{0}, max_late_us{0}, wkc_err{0}, wd_trips{0}, raw_drops{0};
    std::atomic<int> rt_ok{0};
    std::atomic<bool> stop{false};
    std::thread th;
    bool soem_open = false;
    bool closed = false;

    uint8_t* out(int s) { return loopback ? lb_out[s] : ec_slave[s + 1].outputs; }
    uint8_t* in(int s) { return loopback ? lb_in[s] : ec_slave[s + 1].inputs; }

    int motor_for(int s, int ch, uint32_t id) const {
        for (int m = 0; m < int(m_id.size()); ++m)
            if (m_slave[m] == s && m_channel[m] == ch && m_id[m] == id) return m;
        return -1;
    }
};

void put_slot(uint8_t* image, int k, uint32_t id, int dlc, const uint8_t* data, bool rtr = false) {
    uint8_t* p = image + 2 + kSlotBytes * k;
    p[0] = uint8_t(id);
    p[1] = uint8_t(id >> 8);
    p[2] = uint8_t(id >> 16);
    p[3] = uint8_t(id >> 24);
    p[4] = rtr ? 1 : 0;
    p[5] = uint8_t(dlc);
    std::memcpy(p + 6, data, size_t(dlc));
}

// ---- SOEM bring-up --------------------------------------------------------------------------

bool precheck_nic(const char* ifname, char* err, int errlen) {
    // SOEM 1.4.0 ignores the results of socket() and ioctl() and reports every failure as
    // ec_init() == 0, losing errno. Probe the two failures an operator can act on ourselves.
    if (std::strlen(ifname) >= IFNAMSIZ) {
        set_err(err, errlen, "interface name '%s' is too long", ifname);
        return false;
    }
    if (if_nametoindex(ifname) == 0) {
        set_err(err, errlen, "no network interface '%s' (list them with `ip -br link`)", ifname);
        return false;
    }
    int fd = socket(PF_PACKET, SOCK_RAW, htons(kEtherTypeEcat));
    if (fd < 0) {
        int e = errno;
        set_err(err, errlen,
                "cannot open a raw socket on '%s' (%s). EtherCAT needs cap_net_raw: run with sudo, or "
                "grant it (and cap_sys_nice) to this venv's interpreter once with "
                "scripts/grant_ethercat_caps.sh",
                ifname, std::strerror(e));
        return false;
    }
    close(fd);
    return true;
}

bool soem_up(Bus& b, const char* ifname, char* err, int errlen) {
    if (!precheck_nic(ifname, err, errlen)) return false;
    for (int attempt = 1; attempt <= 3; ++attempt) {
        if (attempt > 1) usleep(500000);
        std::memset(b.iomap, 0, sizeof b.iomap);  // every slot empty until the loop writes one
        if (!ec_init(ifname)) {
            set_err(err, errlen, "ec_init(%s) failed", ifname);
            ec_close();
            continue;
        }
        if (ec_config_init(FALSE) <= 0) {
            set_err(err, errlen,
                    "no EtherCAT slaves on '%s': check the cable is in the board's IN port and the link "
                    "has carrier (`ip -br link show %s`)",
                    ifname, ifname);
            ec_close();
            continue;
        }
        int needed = b.raw ? std::min(ec_slavecount, kMaxSlaves) : b.used_slaves;
        if (needed > ec_slavecount) {
            set_err(err, errlen, "configured slave %d, but only %d slave(s) on '%s'", needed - 1, ec_slavecount,
                    ifname);
            ec_close();
            return false;
        }
        for (int i = 1; i <= ec_slavecount; ++i) ec_slave[i].CoEdetails &= ~ECT_COEDET_SDOCA;
        ec_config_map(&b.iomap);
        ec_configdc();  // DC clocks only; no SYNC0: the board runs free
        for (int s = 0; s < needed; ++s) {
            if (ec_slave[s + 1].Obytes < uint32_t(kImageBytes) || ec_slave[s + 1].Ibytes < uint32_t(kImageBytes)) {
                set_err(err, errlen,
                        "slave %d (%s) maps %u output / %u input bytes, need %d each: another EtherCAT master "
                        "is running on '%s' (a second process, or an old bridge), or this is not an "
                        "EtherCAT-to-CAN board of the expected layout",
                        s, ec_slave[s + 1].name, ec_slave[s + 1].Obytes, ec_slave[s + 1].Ibytes, kImageBytes, ifname);
                ec_close();
                return false;
            }
        }
        b.used_slaves = needed;
        ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * 4);
        b.expected_wkc = ec_group[0].outputsWKC * 2 + ec_group[0].inputsWKC;
        ec_slave[0].state = EC_STATE_OPERATIONAL;
        ec_send_processdata();
        ec_receive_processdata(EC_TIMEOUTRET);
        ec_writestate(0);
        for (int i = 0; i < 200 && ec_slave[0].state != EC_STATE_OPERATIONAL; ++i) {
            ec_send_processdata();
            ec_receive_processdata(EC_TIMEOUTRET);
            ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
        }
        if (ec_slave[0].state == EC_STATE_OPERATIONAL) {
            b.soem_open = true;
            return true;
        }
        ec_readstate();
        std::string why;
        for (int i = 1; i <= ec_slavecount; ++i) {
            char line[160];
            snprintf(line, sizeof line, " slave %d state 0x%02x AL 0x%04x (%s);", i - 1, ec_slave[i].state,
                     ec_slave[i].ALstatuscode, ec_ALstatuscode2string(ec_slave[i].ALstatuscode));
            why += line;
        }
        set_err(err, errlen, "EtherCAT did not reach OP on '%s':%s", ifname, why.c_str());
        ec_close();
    }
    return false;
}

// ---- the cycle ------------------------------------------------------------------------------

void place(Bus& b, int s, int slot, uint32_t id, int dlc, const uint8_t* data, bool rtr = false) {
    put_slot(b.out(s), slot, id, dlc, data, rtr);
    b.sent[0][s][slot] = int64_t(id);
}

void exchange_loopback(Bus& b) {
    std::lock_guard<std::mutex> lk(b.mu);  // lb_muted
    for (int s = 0; s < b.used_slaves; ++s) {
        const uint8_t* oldest = b.lb_hist[kReplyLag - 1][s];
        for (int k = 0; k < kSlots; ++k) {
            const uint8_t* q = oldest + 2 + kSlotBytes * k;
            uint32_t id = uint32_t(q[0]) | uint32_t(q[1]) << 8 | uint32_t(q[2]) << 16 | uint32_t(q[3]) << 24;
            bool muted = std::find(b.lb_muted.begin(), b.lb_muted.end(), id) != b.lb_muted.end();
            if (q[5] != 0 && !muted) std::memcpy(b.lb_in[s] + 2 + kSlotBytes * k, q, kSlotBytes);
        }
        for (int d = kReplyLag - 1; d > 0; --d) std::memcpy(b.lb_hist[d][s], b.lb_hist[d - 1][s], kImageBytes);
        std::memcpy(b.lb_hist[0][s], b.lb_out[s], kImageBytes);
    }
}

void tick(Bus& b) {
    for (int d = kReplyLag; d > 0; --d) std::memcpy(b.sent[d], b.sent[d - 1], sizeof b.sent[0]);
    for (auto& row : b.sent[0])
        for (auto& v : row) v = -1;
    for (int s = 0; s < b.used_slaves; ++s) std::memset(b.out(s), 0, kImageBytes);

    {
        std::lock_guard<std::mutex> lk(b.mu);
        if (b.raw) {
            int64_t taken[kMaxSlaves][2][kSlotsPerChannel];  // id placed in each slot this cycle, -1 free
            for (auto& a : taken)
                for (auto& c : a)
                    for (auto& v : c) v = -1;
            for (auto it = b.raw_tx.begin(); it != b.raw_tx.end();) {
                int s = it->slave, c = it->channel - 1;
                int64_t* t = taken[s][c];
                bool dup = std::find(t, t + kSlotsPerChannel, int64_t(it->id)) != t + kSlotsPerChannel;
                int k = it->slot;
                if (k < 0)
                    for (k = 0; k < kSlotsPerChannel && t[k] >= 0; ++k) {
                    }
                if (k < kSlotsPerChannel && t[k] < 0 && !dup) {
                    t[k] = int64_t(it->id);
                    place(b, s, c * kSlotsPerChannel + k, it->id, it->dlc, it->data, it->rtr != 0);
                    it = b.raw_tx.erase(it);
                } else {
                    // Next cycle: its slot is taken, or the board answers none of several same-id frames.
                    ++it;
                }
            }
        } else {
            bool stale = mono_ns() - b.last_set_ns > int64_t(b.watchdog_ms) * 1000000;
            if (stale && !b.tripped) b.wd_trips++;
            b.tripped = stale;
            const std::vector<uint8_t>& src = stale ? b.release : b.tx;
            for (size_t g = 0; g < b.groups.size(); ++g) {
                const std::vector<int>& grp = b.groups[g];
                int count = int(grp.size());
                if (count == 0) continue;
                int s = int(g) / 2, c = int(g) % 2;
                int k_max = std::min(count, kSlotsPerChannel);
                // A rotating group already puts a different motor in each slot every cycle; a group of
                // exactly three is made to by shifting.
                int shift = count == kSlotsPerChannel ? int(b.cycle % kSlotsPerChannel) : 0;
                for (int k = 0; k < k_max; ++k) {
                    int m = grp[(b.ring[g] + k) % count];
                    int slot = c * kSlotsPerChannel + (k + shift) % kSlotsPerChannel;
                    place(b, s, slot, b.m_id[m], 8, &src[size_t(m) * 8]);
                }
                if (count > kSlotsPerChannel) b.ring[g] = (b.ring[g] + kSlotsPerChannel) % count;
            }
        }
    }
    b.cycle++;

    if (b.loopback) {
        exchange_loopback(b);
    } else {
        ec_send_processdata();
        int wkc = ec_receive_processdata(kRxTimeoutUs);
        if (wkc < b.expected_wkc) {
            // The input image may be last cycle's: parse nothing. ponytail: no in-loop recovery; a
            // lost slave shows up as replies that stop arriving, which the host's watchdog catches.
            b.wkc_err++;
            return;
        }
    }

    double stamp = realtime_s();
    std::lock_guard<std::mutex> lk(b.mu);
    if (b.trace.size() < b.trace_want * 2 * kImageBytes && b.used_slaves > 0) {
        b.trace.insert(b.trace.end(), b.out(0), b.out(0) + kImageBytes);
        b.trace.insert(b.trace.end(), b.in(0), b.in(0) + kImageBytes);
    }
    bool pushed = false;
    for (int s = 0; s < b.used_slaves; ++s) {
        const uint8_t* image = b.in(s);
        for (int k = 0; k < kSlots; ++k) {
            const uint8_t* p = image + 2 + kSlotBytes * k;
            int dlc = std::min<int>(p[5], 8);
            uint32_t id = uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
            int ch = k / kSlotsPerChannel + 1;
            if (b.raw) {
                // A reply may land in any slot of its channel, so there is no request to match it to;
                // what identifies a new frame is that the slot's content changed. The cost: an identical
                // reply landing in a slot still holding its twin within ~25 ms is not delivered, and
                // the tool retries -- never a stale reply delivered twice.
                uint8_t* seen = b.raw_seen[s][k];
                if (dlc == 0) {
                    std::memset(seen, 0, kSlotBytes);
                    continue;
                }
                if (std::memcmp(seen, p, kSlotBytes) == 0) continue;
                std::memcpy(seen, p, kSlotBytes);
                RawFrame f{uint8_t(s), uint8_t(ch), uint8_t(dlc), p[4], -1, id, {}, stamp};
                std::memcpy(f.data, p + 6, size_t(dlc));
                if (b.raw_rx.size() >= kRawQueueCap) {
                    b.raw_rx.pop_front();
                    b.raw_drops++;
                }
                b.raw_rx.push_back(f);
                pushed = true;
            } else {
                // Only the motor that was asked: anything else in the slot is an older reply.
                int64_t asked = b.sent[kReplyLag][s][k];
                if (dlc == 0 || asked < 0 || uint32_t(asked) != id) continue;
                int m = b.motor_for(s, ch, id);
                if (m < 0) continue;
                std::memset(&b.rx[size_t(m) * 8], 0, 8);
                std::memcpy(&b.rx[size_t(m) * 8], p + 6, size_t(dlc));
                b.rx_time[size_t(m)] = stamp;
                b.rx_count[size_t(m)]++;
            }
        }
    }
    if (pushed) b.raw_cv.notify_all();
}

void run(Bus* b) {
    if (b->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(b->cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    sched_param sp{};
    sp.sched_priority = b->prio;
    b->rt_ok = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0 ? 1 : 0;

    const int64_t period = int64_t(b->period_us) * 1000;
    int64_t next = mono_ns();
    while (!b->stop.load(std::memory_order_relaxed)) {
        next += period;
        timespec ts{time_t(next / 1000000000), long(next % 1000000000)};
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
        }
        int64_t late = mono_ns() - next;
        if (late > 0 && uint64_t(late / 1000) > b->max_late_us) b->max_late_us = uint64_t(late / 1000);
        if (late > period) {
            // Give the lost cycles up rather than running back-to-back to catch up.
            b->overruns++;
            next += (late / period) * period;
        }
        tick(*b);
        b->ticks++;
    }
}

void close_bus(Bus& b) {
    if (b.closed) return;
    b.closed = true;
    b.stop = true;
    if (b.th.joinable()) b.th.join();
    if (b.soem_open) {
        ec_slave[0].state = EC_STATE_INIT;
        ec_writestate(0);
        ec_close();
        b.soem_open = false;
    }
    b.raw_cv.notify_all();
    g_bus_open = false;
}

}  // namespace

extern "C" {

// Open a bus. n > 0: cyclic mode, `map` is n (slave, channel, can_id) triples and `release` n
// 8-byte frames that are latched until the first ecb_set_frames and whenever the watchdog trips.
// n == 0: raw mode. Returns a handle, or NULL with `err` filled in.
void* ecb_open(const char* ifname, int n, const int* map, int period_us, int watchdog_ms, const uint8_t* release,
               int loopback, int prio, int cpu, char* err, int errlen) {
    if (n < 0 || period_us < 100 || watchdog_ms < 1) {
        set_err(err, errlen, "bad arguments: n=%d period_us=%d watchdog_ms=%d", n, period_us, watchdog_ms);
        return nullptr;
    }
    bool expected = false;
    if (!g_bus_open.compare_exchange_strong(expected, true)) {
        set_err(err, errlen, "an EtherCAT bus is already open in this process (one per process)");
        return nullptr;
    }
    Bus* b = new Bus();
    b->loopback = loopback != 0;
    b->raw = n == 0;
    b->n = n;
    b->period_us = period_us;
    b->watchdog_ms = watchdog_ms;
    b->prio = prio;
    b->cpu = cpu;
    std::memset(b->lb_out, 0, sizeof b->lb_out);
    std::memset(b->lb_in, 0, sizeof b->lb_in);
    std::memset(b->lb_hist, 0, sizeof b->lb_hist);
    std::memset(b->raw_seen, 0, sizeof b->raw_seen);
    for (auto& table : b->sent)
        for (auto& row : table)
            for (auto& v : row) v = -1;

    b->groups.assign(kMaxSlaves * 2, {});
    b->ring.assign(kMaxSlaves * 2, 0);
    for (int m = 0; m < n; ++m) {
        int s = map[3 * m], ch = map[3 * m + 1], id = map[3 * m + 2];
        if (s < 0 || s >= kMaxSlaves || (ch != 1 && ch != 2) || id < 1 || id > 0x7FF) {
            set_err(err, errlen, "motor %d: slave %d channel %d id %d is out of range (slave 0-%d, channel 1-2, id 1-2047)",
                    m, s, ch, id, kMaxSlaves - 1);
            delete b;
            g_bus_open = false;
            return nullptr;
        }
        if (b->motor_for(s, ch, uint32_t(id)) >= 0) {
            set_err(err, errlen, "motor %d: slave %d channel %d id %d is configured twice", m, s, ch, id);
            delete b;
            g_bus_open = false;
            return nullptr;
        }
        b->m_slave.push_back(s);
        b->m_channel.push_back(ch);
        b->m_id.push_back(uint32_t(id));
        b->groups[size_t(s * 2 + ch - 1)].push_back(m);
        b->used_slaves = std::max(b->used_slaves, s + 1);
    }
    for (size_t g = 0; g < b->groups.size(); ++g) {
        size_t count = b->groups[g].size();
        if (count > 0 && count < size_t(kSlotsPerChannel)) {
            set_err(err, errlen,
                    "board %d CAN channel %d has %zu motor(s); it needs at least %d. The board keeps an "
                    "unanswered slot's last reply, and with fewer motors a stopped motor's own last reply "
                    "can sit in its slot and read as fresh",
                    int(g) / 2, int(g) % 2 + 1, count, kSlotsPerChannel);
            delete b;
            g_bus_open = false;
            return nullptr;
        }
    }
    if (b->raw) b->used_slaves = b->loopback ? 1 : 0;  // raw on hardware: every slave found
    b->tx.assign(release, release + size_t(n) * 8);
    b->release.assign(release, release + size_t(n) * 8);
    b->rx.assign(size_t(n) * 8, 0);
    b->rx_time.assign(size_t(n), 0.0);
    b->rx_count.assign(size_t(n), 0);
    b->last_set_ns = mono_ns();

    if (!b->loopback && !soem_up(*b, ifname, err, errlen)) {
        delete b;
        g_bus_open = false;
        return nullptr;
    }

    // The loop thread must never take a signal: SIGINT belongs to Python's main thread, and a signal
    // landing here would cut a sleep or a receive short. Block everything while spawning it so the
    // thread starts with the mask; the calling thread gets its own mask back.
    sigset_t all, old;
    sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &old);
    b->th = std::thread(run, b);
    pthread_sigmask(SIG_SETMASK, &old, nullptr);
    return b;
}

void ecb_set_frames(void* h, const uint8_t* tx) {
    Bus* b = static_cast<Bus*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    std::memcpy(b->tx.data(), tx, b->tx.size());
    b->last_set_ns = mono_ns();
}

void ecb_read(void* h, uint8_t* rx, double* t, uint64_t* count) {
    Bus* b = static_cast<Bus*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    std::memcpy(rx, b->rx.data(), b->rx.size());
    std::memcpy(t, b->rx_time.data(), b->rx_time.size() * sizeof(double));
    std::memcpy(count, b->rx_count.data(), b->rx_count.size() * sizeof(uint64_t));
}

// ticks, overruns, max_late_us, wkc_errors, watchdog_trips, rt_ok, al_state, raw_drops
void ecb_stats(void* h, uint64_t* out) {
    Bus* b = static_cast<Bus*>(h);
    out[0] = b->ticks;
    out[1] = b->overruns;
    out[2] = b->max_late_us;
    out[3] = b->wkc_err;
    out[4] = b->wd_trips;
    out[5] = uint64_t(b->rt_ok.load());
    out[6] = b->loopback ? uint64_t(EC_STATE_OPERATIONAL) : uint64_t(ec_slave[0].state);
    out[7] = b->raw_drops;
}

int ecb_used_slaves(void* h) { return static_cast<Bus*>(h)->used_slaves; }

// Raw mode: queue one frame to go out once, as a remote frame if `rtr`, on channel slot `slot` (0-2)
// or the first free one (-1). 0 queued, -1 bad target, -2 queue full.
int ecb_raw_send(void* h, int slave, int channel, uint32_t id, int dlc, const uint8_t* data, int rtr, int slot) {
    Bus* b = static_cast<Bus*>(h);
    if (!b->raw || slave < 0 || slave >= b->used_slaves || (channel != 1 && channel != 2) || dlc < 0 || dlc > 8 ||
        slot < -1 || slot >= kSlotsPerChannel)
        return -1;
    std::lock_guard<std::mutex> lk(b->mu);
    if (b->raw_tx.size() >= kRawQueueCap) return -2;
    RawFrame f{uint8_t(slave), uint8_t(channel), uint8_t(dlc), uint8_t(rtr != 0), int8_t(slot), id, {}, 0.0};
    std::memcpy(f.data, data, size_t(dlc));
    b->raw_tx.push_back(f);
    return 0;
}

// Raw mode: pop the oldest received frame, waiting up to timeout_us. 1 got one, 0 none.
int ecb_raw_recv(void* h, int timeout_us, int* slave, int* channel, uint32_t* id, int* dlc, uint8_t* data,
                 double* stamp) {
    Bus* b = static_cast<Bus*>(h);
    std::unique_lock<std::mutex> lk(b->mu);
    if (b->raw_rx.empty() && timeout_us > 0)
        b->raw_cv.wait_for(lk, std::chrono::microseconds(timeout_us), [b] { return !b->raw_rx.empty() || b->closed; });
    if (b->raw_rx.empty()) return 0;
    RawFrame f = b->raw_rx.front();
    b->raw_rx.pop_front();
    *slave = f.slave;
    *channel = f.channel;
    *id = f.id;
    *dlc = f.dlc;
    std::memcpy(data, f.data, 8);
    *stamp = f.stamp;
    return 1;
}

// Diagnostics: record board 0's output and input image for the next `cycles` cycles.
void ecb_trace_start(void* h, int cycles) {
    Bus* b = static_cast<Bus*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    b->trace.clear();
    b->trace_want = size_t(std::max(cycles, 0));
}

// Copy out what has been recorded: `out` holds cycles * 2 images (output, then input). Returns cycles.
int ecb_trace_read(void* h, uint8_t* out, int cycles) {
    Bus* b = static_cast<Bus*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    size_t got = std::min(b->trace.size() / (2 * kImageBytes), size_t(std::max(cycles, 0)));
    std::memcpy(out, b->trace.data(), got * 2 * kImageBytes);
    return int(got);
}

// Loopback only: make `id` stop (on != 0) or resume answering, standing in for a dead motor.
void ecb_loopback_mute(void* h, uint32_t id, int on) {
    Bus* b = static_cast<Bus*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    auto it = std::find(b->lb_muted.begin(), b->lb_muted.end(), id);
    if (on && it == b->lb_muted.end()) b->lb_muted.push_back(id);
    if (!on && it != b->lb_muted.end()) b->lb_muted.erase(it);
}

void ecb_close(void* h) {
    Bus* b = static_cast<Bus*>(h);
    close_bus(*b);
    delete b;
}

}  // extern "C"
