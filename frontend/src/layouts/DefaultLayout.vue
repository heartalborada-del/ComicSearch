<script setup lang="ts">
/**
 * Default layout with responsive navigation.
 * - Desktop (md+): Side navigation drawer (rail mode, expandable)
 * - Mobile (<md): Bottom navigation bar
 * - App bar with title and theme toggle button
 */
import { computed, ref } from 'vue'
import { useDisplay } from 'vuetify'
import { useAppTheme } from '@/composables/useTheme'

const display = useDisplay()
const { isDark, toggleTheme } = useAppTheme()

const drawer = ref(false)

interface NavItem {
  icon: string
  title: string
  to: string
}

const navItems: NavItem[] = [
  { icon: 'mdi-image-search', title: '搜索', to: '/' },
  { icon: 'mdi-import', title: '导入', to: '/import' },
  { icon: 'mdi-format-list-checks', title: '任务', to: '/tasks' },
]

const isMobile = computed(() => display.mdAndDown.value)
</script>

<template>
  <v-app>
    <!-- App Bar -->
    <v-app-bar flat elevation="0" class="bg-surface-container">
      <template v-if="!isMobile" #prepend>
        <v-app-bar-nav-icon
          :icon="drawer ? 'mdi-menu-open' : 'mdi-menu'"
          @click="drawer = !drawer"
        />
      </template>

      <v-app-bar-title class="font-weight-medium">
        <v-icon start color="primary" size="28">mdi-book-search-outline</v-icon>
        ComicSearch
      </v-app-bar-title>

      <template #append>
        <v-btn
          :icon="isDark ? 'mdi-weather-night' : 'mdi-white-balance-sunny'"
          variant="text"
          @click="toggleTheme"
        />
      </template>
    </v-app-bar>

    <!-- Desktop: Side Navigation Drawer -->
    <v-navigation-drawer
      v-if="!isMobile"
      v-model="drawer"
      :rail="!drawer"
      rail-width="68"
      width="240"
      class="bg-surface-container"
      floating
    >
      <v-list nav density="comfortable">
        <v-list-item
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          :prepend-icon="item.icon"
          :title="item.title"
          rounded="lg"
          color="primary"
          class="mb-1"
        />
      </v-list>
    </v-navigation-drawer>

    <!-- Main Content -->
    <v-main>
      <v-container class="pa-3 pa-sm-4 pa-md-6" :max-width="1400">
        <router-view v-slot="{ Component }">
          <transition name="fade-transition" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </v-container>
    </v-main>

    <!-- Mobile: Bottom Navigation -->
    <v-bottom-navigation
      v-if="isMobile"
      color="primary"
      grow
      elevation="0"
    >
      <v-btn
        v-for="item in navItems"
        :key="item.to"
        :to="item.to"
        :prepend-icon="item.icon"
      >
        {{ item.title }}
      </v-btn>
    </v-bottom-navigation>
  </v-app>
</template>
