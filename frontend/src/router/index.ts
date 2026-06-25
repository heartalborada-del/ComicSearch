/**
 * Vue Router configuration with auth guards.
 * Protected routes redirect to login when auth is enabled and user is not logged in.
 */
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const routes: RouteRecordRaw[] = [
    {
        path: '/',
        name: 'search',
        component: () => import('@/pages/SearchPage.vue'),
        meta: { title: '搜索', requiresAuth: false },
    },
    {
        path: '/info/:id',
        name: 'detail',
        component: () => import('@/pages/DetailPage.vue'),
        props: true,
        meta: { title: '漫画详情', requiresAuth: false },
    },
    {
        path: '/import',
        name: 'import',
        component: () => import('@/pages/ImportPage.vue'),
        meta: { title: 'E-Hentai 导入', requiresAuth: true },
    },
    {
        path: '/tasks',
        name: 'tasks',
        component: () => import('@/pages/TasksPage.vue'),
        meta: { title: '任务管理', requiresAuth: true },
    },
    {
        path: '/login',
        name: 'login',
        component: () => import('@/pages/LoginPage.vue'),
        meta: { title: '登录' },
    },
]

const router = createRouter({
    history: createWebHistory(),
    routes,
    scrollBehavior(_to, _from, savedPosition) {
        return savedPosition ?? { top: 0 }
    },
})

// Global auth guard: redirect to login if route requires auth and
// the user is not logged in while auth is enabled.
router.beforeEach(async (to, _from, next) => {
    // We need to access the auth store. It's initialized in main.ts,
    // so just call useAuthStore().
    const authStore = useAuthStore()

    // If auth hasn't been checked yet, check it now
    if (!authStore.authEnabled && !authStore.loading) {
        await authStore.checkAuthStatus()
    }

    // If the route doesn't require auth, allow
    if (!to.meta.requiresAuth) {
        next()
        return
    }

    // If auth is not enabled on server, allow
    if (!authStore.authEnabled) {
        next()
        return
    }

    // If logged in, allow
    if (authStore.loggedIn) {
        next()
        return
    }

    // Redirect to login with return URL
    next({ name: 'login', query: { redirect: to.fullPath } })
})

router.afterEach((to) => {
    const title = (to.meta.title as string) || ''
    document.title = title ? `${title} - ComicSearch` : 'ComicSearch'
})

export default router
