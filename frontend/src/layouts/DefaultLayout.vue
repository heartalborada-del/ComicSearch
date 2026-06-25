<script setup lang="ts">
/**
 * Default layout with responsive navigation.
 * - Desktop (md+): Side navigation drawer (rail mode, expandable)
 * - Mobile (<md): Bottom navigation bar
 * - App bar with title, user menu, quota display, and theme toggle
 */
import { computed, ref, watch } from 'vue'
import { useDisplay } from 'vuetify'
import { useRouter } from 'vue-router'
import { useAppTheme } from '@/composables/useTheme'
import { useAuthStore } from '@/stores/auth'

const display = useDisplay()
const router = useRouter()
const { isDark, toggleTheme } = useAppTheme()
const authStore = useAuthStore()

const drawer = ref(false)

interface NavItem {
  icon: string
  title: string
  to: string
  requiresAuth?: boolean
  adminOnly?: boolean
}

const navItems: NavItem[] = [
  { icon: 'mdi-image-search', title: '搜索', to: '/' },
  { icon: 'mdi-tag-search', title: '标签', to: '/tags', requiresAuth: true },
  { icon: 'mdi-import', title: '导入', to: '/import', requiresAuth: true },
  { icon: 'mdi-format-list-checks', title: '任务', to: '/tasks', requiresAuth: true },
  { icon: 'mdi-shield-account', title: '管理', to: '/admin', requiresAuth: true, adminOnly: true },
]

/** Filter nav items: show auth-required items when logged in, admin items only for admins. */
const visibleNavItems = computed(() =>
  navItems.filter(
    (item) => {
      if (item.adminOnly && !authStore.isAdmin) return false
      if (item.requiresAuth && authStore.authEnabled && !authStore.loggedIn) return false
      return true
    },
  ),
)

const isMobile = computed(() => display.mdAndDown.value)

const quotaText = computed(() => {
  if (!authStore.authEnabled || !authStore.loggedIn) return null
  if (authStore.quotaUnlimited) return '无限'
  return `${authStore.quotaRemaining}/${authStore.quotaTotal}`
})

function goToLogin(): void {
  router.push({ name: 'login', query: { redirect: router.currentRoute.value.fullPath } })
}

function handleLogout(): void {
  authStore.logout()
  router.push({ name: 'search' })
}

// Watch auth state to refresh quota when logging in
watch(
  () => authStore.loggedIn,
  (newVal) => {
    if (newVal) {
      authStore.fetchQuota()
    }
  },
)
</script>

<template>
  <v-app>
    <!-- App Bar -->
    <v-app-bar flat elevation="0" class="bg-surface-container">
      <template v-if="!isMobile" #prepend>
        <v-app-bar-nav-icon :icon="drawer ? 'mdi-menu-open' : 'mdi-menu'" @click="drawer = !drawer" />
      </template>

      <v-app-bar-title class="font-weight-medium">
        <v-icon start color="primary" size="28">mdi-book-search-outline</v-icon>
        ComicSearch
      </v-app-bar-title>

      <template #append>
        <!-- Quota Chip (shown when logged in and auth enabled) -->
        <v-chip v-if="authStore.authEnabled && authStore.loggedIn && quotaText && !isMobile" size="small"
          variant="tonal" color="primary" class="mr-2">
          <v-icon start size="14">mdi-calendar-today</v-icon>
          {{ quotaText }}
        </v-chip>

        <!-- User Menu (authenticated) -->
        <v-menu v-if="authStore.authEnabled && authStore.loggedIn" offset-y>
          <template #activator="{ props: menuProps }">
            <v-btn v-bind="menuProps" variant="text" size="small" rounded="lg" class="mr-1">
              <v-icon start size="20">mdi-account-circle</v-icon>
              <span class="d-none d-sm-inline">{{ authStore.user?.username }}</span>
            </v-btn>
          </template>
          <v-list density="compact" min-width="180" rounded="lg">
            <v-list-item>
              <v-list-item-title class="font-weight-medium">
                {{ authStore.user?.username }}
              </v-list-item-title>
              <v-list-item-subtitle v-if="authStore.isAdmin">
                管理员
              </v-list-item-subtitle>
            </v-list-item>
            <v-divider />
            <v-list-item v-if="authStore.quota" density="compact">
              <template #prepend>
                <v-icon size="18" color="primary">mdi-calendar-today</v-icon>
              </template>
              <v-list-item-title class="text-body-2">
                今日配额: {{ quotaText }}
              </v-list-item-title>
              <v-list-item-subtitle v-if="authStore.quotaResetRelative" class="text-caption">
                {{ authStore.quotaResetRelative }}
              </v-list-item-subtitle>
            </v-list-item>
            <v-list-item density="compact" @click="handleLogout">
              <template #prepend>
                <v-icon size="18" color="error">mdi-logout</v-icon>
              </template>
              <v-list-item-title class="text-body-2">退出登录</v-list-item-title>
            </v-list-item>
          </v-list>
        </v-menu>

        <!-- Login Button (unauthenticated, auth enabled) -->
        <v-btn v-if="authStore.authEnabled && !authStore.loggedIn" variant="tonal" size="small" rounded="lg"
          prepend-icon="mdi-login" class="mr-1" @click="goToLogin">
          登录
        </v-btn>

        <v-btn :icon="isDark ? 'mdi-weather-night' : 'mdi-white-balance-sunny'" variant="text" @click="toggleTheme" />
      </template>
    </v-app-bar>

    <!-- Desktop: Side Navigation Drawer -->
    <v-navigation-drawer v-if="!isMobile" v-model="drawer" :rail="!drawer" rail-width="68" width="240"
      class="bg-surface-container" floating>
      <v-list nav density="comfortable">
        <v-list-item v-for="item in visibleNavItems" :key="item.to" :to="item.to" :prepend-icon="item.icon"
          :title="item.title" rounded="lg" color="primary" class="mb-1" />
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
    <v-bottom-navigation v-if="isMobile" color="primary" grow elevation="0">
      <v-btn v-for="item in visibleNavItems" :key="item.to" :to="item.to" :prepend-icon="item.icon">
        {{ item.title }}
      </v-btn>
    </v-bottom-navigation>
  </v-app>
</template>
