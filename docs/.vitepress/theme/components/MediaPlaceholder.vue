<template>
  <div :class="['media-placeholder', type === 'video' ? 'video-placeholder' : 'photo-placeholder']">
    <div class="placeholder-icon">{{ type === 'video' ? '🎬' : '📷' }}</div>
    <div class="placeholder-label">{{ type === 'video' ? 'Video' : 'Photo' }} — Coming Soon</div>
    <div class="placeholder-desc">{{ description }}</div>
  </div>
</template>

<script setup>
defineProps({
  type: { type: String, default: 'video' }, // 'video' | 'photo'
  description: { type: String, default: 'Media asset placeholder — will be filmed/photographed and added here.' },
})
</script>
