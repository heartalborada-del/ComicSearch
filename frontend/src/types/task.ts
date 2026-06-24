/**
 * Type definitions for the task API.
 */

/** Task status values. */
export type TaskStatus = 'pending' | 'running' | 'success' | 'failed'

/** Task type identifier. */
export type TaskType = 'ehentai_import' | string

/** Single task record returned by GET /tasks/{id} and GET /tasks. */
export interface TaskRecord {
    task_id: string
    task_type: TaskType
    status: TaskStatus
    cancel_requested: boolean
    created_at: string
    started_at: string | null
    finished_at: string | null
    result: Record<string, unknown> | null
    error: string | null
}

/** Request body for POST /ehentai/import/tasks. */
export interface SubmitImportRequest {
    url?: string
    urls?: string[]
    crop_faces?: boolean
}

/** Per-URL result in the import task submission response. */
export interface SubmitImportItem {
    url: string
    task_id: string
    status: string
    is_duplicate: boolean
}

/** Response from POST /ehentai/import/tasks. */
export interface SubmitImportResponse {
    task_id: string | null
    status: string | null
    items: SubmitImportItem[]
}

/** Query parameters for GET /tasks. */
export interface ListTasksParams {
    limit?: number
    status?: TaskStatus
}
