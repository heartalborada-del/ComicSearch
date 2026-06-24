/**
 * Vue Router configuration with four routes.
 * Scroll behavior restores position on back/forward navigation.
 */
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
    {
        path: '/',
        name: 'search',
        component: () => import('@/pages/SearchPage.vue'),
        meta: { title: '搜索' },
    },
    {
        path: '/info/:id',
        name: 'detail',
        component: () => import('@/pages/DetailPage.vue'),
        props: true,
        meta: { title: '漫画详情' },
    },
    {
        path: '/import',
        name: 'import',
        component: () => import('@/pages/ImportPage.vue'),
        meta: { title: 'E-Hentai 导入' },
    },
    {
        path: '/tasks',
        name: 'tasks',
        component: () => import('@/pages/TasksPage.vue'),
        meta: { title: '任务管理' },
    },
]

const router = createRouter({
    history: createWebHistory(),
    routes,
    scrollBehavior(_to, _from, savedPosition) {
        return savedPosition ?? { top: 0 }
    },
})

router.afterEach((to) => {
    const title = (to.meta.title as string) || ''
    document.title = title ? `${title} - ComicSearch` : 'ComicSearch'
})

export default router
