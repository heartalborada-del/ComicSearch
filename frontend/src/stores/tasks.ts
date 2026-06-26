/**
 * Tasks store — manages task list, polling for active tasks,
 * and task submission/cancellation.
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { cancelTask, getTask, listTasks, submitImport } from '@/api/tasks'
import { ApiError } from '@/api/client'
import type {
    ListTasksParams,
    SubmitImportRequest,
    SubmitImportResponse,
    TaskRecord,
    TaskStatus,
} from '@/types/task'

/** Polling interval in milliseconds. */
const POLL_INTERVAL_MS = 3000
/** Tasks per page. */
const PAGE_SIZE = 20

export const useTasksStore = defineStore('tasks', () => {
    const tasks = ref<TaskRecord[]>([])
    const loading = ref(false)
    const error = ref<string | null>(null)
    const submitting = ref(false)
    const submitResult = ref<SubmitImportResponse | null>(null)
    const submitError = ref<string | null>(null)

    /** Server-side status filter. */
    const statusFilter = ref<TaskStatus | undefined>(undefined)

    /** Pagination state. */
    const totalCount = ref(0)
    const currentPage = ref(1)

    /** Total pages based on server-reported total. */
    const totalPages = computed(() => Math.max(1, Math.ceil(totalCount.value / PAGE_SIZE)))

    /** Whether polling is active. */
    const polling = ref(false)
    let pollTimer: ReturnType<typeof setInterval> | null = null

    /** Tasks are already filtered server-side; this alias exists for backward compat. */
    const filteredTasks = computed(() => tasks.value)

    /** Whether any tasks are pending or running. */
    const hasActiveTasks = computed(() =>
        tasks.value.some((t) => t.status === 'pending' || t.status === 'running'),
    )

    /** Count tasks by status (uses server-reported total for 'all'). */
    const statusCounts = computed(() => {
        const counts: Record<string, number> = {
            all: totalCount.value,
            pending: 0,
            running: 0,
            success: 0,
            failed: 0,
        }
        for (const task of tasks.value) {
            counts[task.status] = (counts[task.status] || 0) + 1
        }
        return counts
    })

    /**
     * Fetch the task list from the server for the current page.
     */
    async function fetchTasks(): Promise<void> {
        loading.value = true
        error.value = null

        const params: ListTasksParams = {
            limit: PAGE_SIZE,
            offset: (currentPage.value - 1) * PAGE_SIZE,
        }
        if (statusFilter.value) {
            params.status = statusFilter.value
        }

        try {
            const response = await listTasks(params)
            tasks.value = response.items
            totalCount.value = response.total
        } catch (err) {
            error.value = err instanceof ApiError ? err.detail : '获取任务列表失败'
        } finally {
            loading.value = false
        }
    }

    /**
     * Change the current page and re-fetch.
     */
    function goToPage(page: number): void {
        if (page < 1 || page > totalPages.value) return
        currentPage.value = page
        fetchTasks()
    }

    /**
     * Refresh only active (pending/running) tasks.
     * Updates individual task records without replacing the full list.
     */
    async function refreshActiveTasks(): Promise<void> {
        const activeTasks = tasks.value.filter((t) => t.status === 'pending' || t.status === 'running')
        if (activeTasks.length === 0) {
            stopPolling()
            return
        }

        try {
            const updated = await Promise.all(activeTasks.map((t) => getTask(t.task_id)))
            const updatedMap = new Map(updated.map((t) => [t.task_id, t]))
            tasks.value = tasks.value.map((t) => updatedMap.get(t.task_id) ?? t)
        } catch {
            // Silently ignore polling errors
        }
    }

    /**
     * Start polling for active task updates.
     */
    function startPolling(): void {
        if (polling.value) return
        polling.value = true
        pollTimer = setInterval(() => {
            if (document.visibilityState === 'visible') {
                refreshActiveTasks()
            }
        }, POLL_INTERVAL_MS)
    }

    /**
     * Stop polling.
     */
    function stopPolling(): void {
        polling.value = false
        if (pollTimer) {
            clearInterval(pollTimer)
            pollTimer = null
        }
    }

    /**
     * Submit an E-Hentai import task.
     */
    async function submitImportTask(request: SubmitImportRequest): Promise<void> {
        submitting.value = true
        submitError.value = null
        submitResult.value = null

        try {
            submitResult.value = await submitImport(request)
            // Refresh task list after submission
            await fetchTasks()
            startPolling()
        } catch (err) {
            submitError.value = err instanceof ApiError ? err.detail : '提交导入任务失败'
        } finally {
            submitting.value = false
        }
    }

    /**
     * Cancel a task by ID.
     */
    async function cancelTaskById(taskId: string): Promise<void> {
        try {
            const updated = await cancelTask(taskId)
            tasks.value = tasks.value.map((t) => (t.task_id === taskId ? updated : t))
        } catch (err) {
            error.value = err instanceof ApiError ? err.detail : '取消任务失败'
        }
    }

    /**
     * Set the status filter, reset to page 1, and re-fetch.
     */
    function setStatusFilter(status: TaskStatus | undefined): void {
        statusFilter.value = status
        currentPage.value = 1
        fetchTasks()
    }

    return {
        tasks,
        loading,
        error,
        submitting,
        submitResult,
        submitError,
        statusFilter,
        polling,
        filteredTasks,
        hasActiveTasks,
        statusCounts,
        totalCount,
        currentPage,
        totalPages,
        PAGE_SIZE,
        fetchTasks,
        goToPage,
        refreshActiveTasks,
        startPolling,
        stopPolling,
        submitImportTask,
        cancelTaskById,
        setStatusFilter,
    }
})
