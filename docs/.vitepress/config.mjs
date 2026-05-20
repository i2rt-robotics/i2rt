import { defineConfig } from 'vitepress'

const base = process.env.DEPLOY_TARGET === 'cloudflare' ? '/' : '/JH_i2rt/'

export default defineConfig({
  title: 'I2RT Robotics',
  description: 'Open robotics platform for research and embodied AI — YAM arms, Flow Base, and more.',
  base,
  appearance: true,

  // Force no-cache headers in local preview so changes are always visible after rebuild
  vite: {
    preview: {
      headers: {
        'Cache-Control': 'no-store',
      },
    },
  },

  head: [
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    // Platypi: i2rt.com's heading font. IBM Plex Mono: code blocks.
    ['link', { href: 'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Platypi:ital,wght@0,400;0,500;0,600;1,400&display=swap', rel: 'stylesheet' }],
    ['meta', { name: 'theme-color', content: '#855832' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'I2RT Robotics Docs' }],
    ['meta', { property: 'og:description', content: 'Documentation for the YAM arm family, Flow Base, and Linear Bot.' }],
  ],

  themeConfig: {
    siteTitle: 'I2RT Docs',

    nav: [
      { text: 'Get Started', link: '/getting-started/sw-setup' },
      { text: 'GitHub', link: 'https://github.com/i2rt-robotics/i2rt', target: '_blank' },
      { text: 'i2rt.com', link: 'https://i2rt.com', target: '_blank' },
    ],

    sidebar: {
      // Use the same sidebar for /getting-started/ and /products/ so users can
      // navigate between SW Setup and product pages without changing context.
      '/': [
        {
          text: 'Get Started',
          items: [
            { text: 'SW Setup', link: '/getting-started/sw-setup' },
          ],
        },
        {
          text: 'Products',
          items: [
            { text: 'Product Family Overview', link: '/products/' },
            { text: 'YAM Arm Series', link: '/products/yam' },
            { text: 'YAM Arm', link: '/products/yam-arm' },
            { text: 'YAM Leader Arm', link: '/products/yam-leader' },
            { text: 'YAM Cell', link: '/products/yam-cell' },
            { text: 'YAM Box', link: '/products/yam-box' },
            { text: 'Flow Base', link: '/products/flow-base' },
            { text: 'Linear Bot', link: '/products/linear-bot' },
            { text: 'Motors', link: '/products/motors' },
          ],
        },
        {
          text: 'Motor Details',
          collapsed: true,
          items: [
            { text: 'GF43X10-10', link: '/products/motor-gf43x10-10' },
            { text: 'GF43X40-10', link: '/products/motor-gf43x40-10' },
            { text: 'GF43X40-16', link: '/products/motor-gf43x40-16' },
            { text: 'Rovomotor ↗', link: 'https://rovomotor.com', target: '_blank' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/i2rt-robotics/i2rt' },
      {
        icon: {
          svg: '<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057.11 18.101.127 18.143.158 18.17a19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03z"/></svg>',
        },
        link: 'https://discord.gg/i2rt',
      },
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 I2RT Robotics, LLC — Fremont, CA',
    },

    search: { provider: 'local' },

    editLink: {
      pattern: 'https://github.com/i2rt-robotics/i2rt/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },
  },
})
