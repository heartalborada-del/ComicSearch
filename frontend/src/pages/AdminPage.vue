<script setup lang="ts">
/**
 * Admin page — user management + import review.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { getJson, postJson, ApiError } from '@/api/client'
import type { TaskRecord } from '@/types/task'
import { listReviewTasks, approveTask, rejectTask } from '@/api/tasks'
import { resetUserPassword } from '@/api/auth'
import { useAuthStore } from '@/stores/auth'

// ---- Types ----
interface AdminUserItem {
    id: number
    username: string
    is_admin: boolean
    is_active: boolean
    created_at: string
    daily_quota: number | null
    used_today: number
    registration_ip: string | null
    last_login_ips: string[]
}

const authStore = useAuthStore()

// ---- Tabs ----
const activeTab = ref<'users' | 'review'>('users')

// ---- State (users) ----
const allUsers = ref<AdminUserItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)
const searchQuery = ref('')
const filterStatus = ref<'all' | 'active' | 'banned'>('all')
const filterRole = ref<'all' | 'admin' | 'user'>('all')
const selectedIds = ref<Set<number>>(new Set())
const bulkLoading = ref(false)
const bulkQuotaDialog = ref(false)
const bulkQuota = ref<number>(0)
const banSaving = ref(false)

// ---- Edit User Dialog ----
const editDialog = ref(false)
const editUserId = ref<number>(0)
const editUsername = ref('')
const editQuota = ref<number>(0)
const editNewPassword = ref('')
const editConfirmPassword = ref('')
const editLoading = ref(false)
const editError = ref<string | null>(null)
const toast = ref<{ type: 'success' | 'error'; text: string } | null>(null)

const globalDefaultQuota = computed(() => authStore.quota?.daily_quota ?? 50)

// ---- State (review) ----
const reviewTasks = ref<TaskRecord[]>([])
const reviewLoading = ref(false)
const reviewActing = ref(false)

// ---- Computed ----
const filteredUsers = computed(() => {
    let list = allUsers.value
    if (searchQuery.value.trim()) {
        const q = searchQuery.value.trim().toLowerCase()
        list = list.filter(
            (u) =>
                u.username.toLowerCase().includes(q) ||
                String(u.id).includes(q) ||
                (u.registration_ip && u.registration_ip.includes(q)) ||
                u.last_login_ips.some((ip) => ip.includes(q)),
        )
    }
    if (filterStatus.value === 'active') list = list.filter((u) => u.is_active)
    else if (filterStatus.value === 'banned') list = list.filter((u) => !u.is_active)
    if (filterRole.value === 'admin') list = list.filter((u) => u.is_admin)
    else if (filterRole.value === 'user') list = list.filter((u) => !u.is_admin)
    return list
})

const selectedCount = computed(
    () => filteredUsers.value.filter((u) => selectedIds.value.has(u.id)).length,
)
const isAllSelected = computed(
    () =>
        filteredUsers.value.length > 0 &&
        filteredUsers.value.every((u) => selectedIds.value.has(u.id)),
)

function showToast(type: 'success' | 'error', text: string): void {
    toast.value = { type, text }
    setTimeout(() => { toast.value = null }, 3000)
}

// ---- Actions ----
async function loadUsers(): Promise<void> {
    loading.value = true
    error.value = null
    try {
        allUsers.value = await getJson<AdminUserItem[]>('/auth/users')
    } catch (err) {
        error.value = err instanceof ApiError ? err.detail : '加载用户列表失败'
    } finally {
        loading.value = false
    }
}

// --- Selection ---
function toggleSelectAll(): void {
    if (isAllSelected.value) {
        selectedIds.value = new Set()
    } else {
        selectedIds.value = new Set(filteredUsers.value.map((u) => u.id))
    }
}

function toggleSelect(id: number): void {
    const next = new Set(selectedIds.value)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    selectedIds.value = next
}

watch([searchQuery, filterStatus, filterRole], () => {
    selectedIds.value = new Set()
})

// --- Single user operations ---
function openEditUser(user: AdminUserItem): void {
    editUserId.value = user.id
    editUsername.value = user.username
    editQuota.value = user.daily_quota ?? globalDefaultQuota.value
    editNewPassword.value = ''
    editConfirmPassword.value = ''
    editError.value = null
    editDialog.value = true
}

function closeEditUser(): void {
    editDialog.value = false
    editUserId.value = 0
    editUsername.value = ''
    editError.value = null
}

async function doSaveEditUser(): Promise<void> {
    editError.value = null

    // Validate password if provided
    const pwd = editNewPassword.value.trim()
    if (pwd && pwd.length < 6) {
        editError.value = '密码至少需要 6 个字符'
        return
    }
    if (pwd && pwd !== editConfirmPassword.value) {
        editError.value = '两次输入的密码不一致'
        return
    }

    editLoading.value = true
    const parts: string[] = []
    try {
        // Always update quota
        await postJson<{ count: number }>('/auth/quota/set', {
            user_ids: [editUserId.value],
            daily_quota: editQuota.value,
        })
        const u = allUsers.value.find((x) => x.id === editUserId.value)
        if (u) {
            u.daily_quota = editQuota.value <= 0 ? null : editQuota.value
        }
        parts.push('配额已更新')

        // Only reset password if provided
        if (pwd) {
            await resetUserPassword({
                user_id: editUserId.value,
                new_password: pwd,
            })
            parts.push('密码已重置')
        }

        const username = editUsername.value
        closeEditUser()
        showToast('success', `${username} ${parts.join('，')}`)
    } catch (err) {
        editError.value = err instanceof ApiError ? err.detail : '操作失败'
    } finally {
        editLoading.value = false
    }
}

async function toggleBan(user: AdminUserItem): Promise<void> {
    banSaving.value = true
    const endpoint = user.is_active ? '/auth/users/ban' : '/auth/users/unban'
    try {
        await postJson<{ count: number }>(endpoint, { user_ids: [user.id] })
        user.is_active = !user.is_active
        showToast('success', `${user.username} 已${user.is_active ? '解禁' : '封禁'}`)
    } catch (err) {
        showToast('error', err instanceof ApiError ? err.detail : '操作失败')
    } finally {
        banSaving.value = false
    }
}

// --- Bulk operations ---
async function bulkBan(): Promise<void> {
    if (selectedCount.value === 0) return
    bulkLoading.value = true
    try {
        const r = await postJson<{ count: number }>('/auth/users/ban', {
            user_ids: [...selectedIds.value],
        })
        for (const id of selectedIds.value) {
            const u = allUsers.value.find((x) => x.id === id)
            if (u) u.is_active = false
        }
        selectedIds.value = new Set()
        showToast('success', `已封禁 ${r.count} 个用户`)
    } catch (err) {
        showToast('error', err instanceof ApiError ? err.detail : '批量封禁失败')
    } finally {
        bulkLoading.value = false
    }
}

async function bulkUnban(): Promise<void> {
    if (selectedCount.value === 0) return
    bulkLoading.value = true
    try {
        const r = await postJson<{ count: number }>('/auth/users/unban', {
            user_ids: [...selectedIds.value],
        })
        for (const id of selectedIds.value) {
            const u = allUsers.value.find((x) => x.id === id)
            if (u) u.is_active = true
        }
        selectedIds.value = new Set()
        showToast('success', `已解禁 ${r.count} 个用户`)
    } catch (err) {
        showToast('error', err instanceof ApiError ? err.detail : '批量解禁失败')
    } finally {
        bulkLoading.value = false
    }
}

function openBulkQuota(): void {
    bulkQuota.value = 0
    bulkQuotaDialog.value = true
}

async function doBulkQuota(): Promise<void> {
    bulkLoading.value = true
    try {
        const r = await postJson<{ count: number }>('/auth/quota/set', {
            user_ids: [...selectedIds.value],
            daily_quota: bulkQuota.value,
        })
        await loadUsers()
        selectedIds.value = new Set()
        bulkQuotaDialog.value = false
        showToast('success', `已更新 ${r.count} 个用户的配额`)
    } catch (err) {
        showToast('error', err instanceof ApiError ? err.detail : '批量设置失败')
        bulkQuotaDialog.value = false
    } finally {
        bulkLoading.value = false
    }
}

// --- Helpers ---
function quotaLabel(quota: number | null): string {
    if (quota === null || quota === 0) return `全局 (${globalDefaultQuota.value})`
    if (quota < 0) return '无限'
    return String(quota)
}

function roleChip(user: AdminUserItem): { color: string; text: string } {
    if (user.is_admin) return { color: 'primary', text: '管理员' }
    if (!user.is_active) return { color: 'error', text: '已封禁' }
    return { color: '', text: '用户' }
}

// --- Review actions ---
async function loadReviewTasks(): Promise<void> {
    reviewLoading.value = true
    try {
        reviewTasks.value = await listReviewTasks()
    } catch (err) {
        showToast('error', err instanceof ApiError ? err.detail : '加载审核列表失败')
    } finally {
        reviewLoading.value = false
    }
}

async function doApprove(task: TaskRecord): Promise<void> {
    reviewActing.value = true
    try {
        await approveTask(task.task_id)
        reviewTasks.value = reviewTasks.value.filter((t) => t.task_id !== task.task_id)
        showToast('success', '已通过审核')
    } catch (err) {
        showToast('error', err instanceof ApiError ? err.detail : '审核失败')
    } finally {
        reviewActing.value = false
    }
}

async function doReject(task: TaskRecord): Promise<void> {
    reviewActing.value = true
    try {
        await rejectTask(task.task_id)
        reviewTasks.value = reviewTasks.value.filter((t) => t.task_id !== task.task_id)
        showToast('success', '已拒绝')
    } catch (err) {
        showToast('error', err instanceof ApiError ? err.detail : '拒绝失败')
    } finally {
        reviewActing.value = false
    }
}

function parseTaskUrl(task: TaskRecord): string {
    const p = task.payload as Record<string, unknown> | null
    return (p?.url as string) ?? '—'
}

function reviewStatusChip(task: TaskRecord): { color: string; text: string } {
    const statusMap: Record<string, { color: string; text: string }> = {
        pending_review: { color: 'warning', text: '待审核' },
        pending: { color: 'info', text: '等待执行' },
        running: { color: 'primary', text: '执行中' },
        success: { color: 'success', text: '已完成' },
        failed: { color: 'error', text: '失败' },
        cancelled: { color: '', text: '已取消' },
    }
    return statusMap[task.status] ?? { color: '', text: task.status }
}

watch(activeTab, (tab) => {
    if (tab === 'review') loadReviewTasks()
})

onMounted(loadUsers)
</script>

<template>
    <div class="admin-page">
        <!-- Header + Tabs -->
        <div class="d-flex align-center ga-3 mb-4 flex-wrap">
            <h1 class="text-h5 font-weight-medium">管理后台</h1>
            <v-btn-toggle v-model="activeTab" mandatory density="comfortable" variant="tonal" color="primary">
                <v-btn value="users" size="small" prepend-icon="mdi-account-group">
                    用户管理
                </v-btn>
                <v-btn value="review" size="small" prepend-icon="mdi-file-document-check" class="ml-2">
                    导入审核
                    <v-badge v-if="reviewTasks.length > 0" :content="reviewTasks.length" color="warning" inline
                        class="ml-1" />
                </v-btn>
            </v-btn-toggle>
            <v-chip v-if="allUsers.length && activeTab === 'users'" size="small" variant="tonal" color="primary">
                {{ allUsers.length }} 个用户
            </v-chip>
            <v-chip v-if="selectedCount > 0 && activeTab === 'users'" size="small" variant="tonal" color="secondary">
                已选 {{ selectedCount }}
            </v-chip>
        </div>

        <!-- Toast -->
        <v-alert v-if="toast" :type="toast.type" variant="tonal" rounded="lg" class="mb-3" closable density="compact">
            {{ toast.text }}
        </v-alert>

        <!-- ====== User Management Tab ====== -->
        <div class="tab-panel" :class="{ 'tab-panel--active': activeTab === 'users' }">

            <!-- Toolbar -->
            <v-card variant="tonal" class="bg-surface-container mb-4" rounded="lg">
                <v-card-text class="pa-3">
                    <v-row dense align="center">
                        <v-col cols="12" sm="4" md="3">
                            <v-text-field v-model="searchQuery" label="搜索用户名/ID/IP" density="compact" variant="outlined"
                                hide-details clearable prepend-inner-icon="mdi-magnify" />
                        </v-col>
                        <v-col cols="6" sm="3" md="2">
                            <v-select v-model="filterStatus" :items="[
                                { title: '全部状态', value: 'all' },
                                { title: '正常', value: 'active' },
                                { title: '已封禁', value: 'banned' },
                            ]" density="compact" variant="outlined" hide-details />
                        </v-col>
                        <v-col cols="6" sm="3" md="2">
                            <v-select v-model="filterRole" :items="[
                                { title: '全部角色', value: 'all' },
                                { title: '管理员', value: 'admin' },
                                { title: '用户', value: 'user' },
                            ]" density="compact" variant="outlined" hide-details />
                        </v-col>
                        <v-col cols="12" sm="2" md="5">
                            <div class="d-flex ga-1 flex-wrap justify-sm-end">
                                <v-btn v-if="selectedCount > 0" size="small" variant="tonal" color="error"
                                    :loading="bulkLoading" @click="bulkBan">
                                    批量封禁
                                </v-btn>
                                <v-btn v-if="selectedCount > 0" size="small" variant="tonal" color="success"
                                    :loading="bulkLoading" @click="bulkUnban">
                                    批量解禁
                                </v-btn>
                                <v-btn v-if="selectedCount > 0" size="small" variant="tonal" color="primary"
                                    :loading="bulkLoading" @click="openBulkQuota">
                                    批量设配额
                                </v-btn>
                                <span v-if="selectedCount === 0" class="text-caption text-on-surface-variant">
                                    勾选用户后可批量操作
                                </span>
                            </div>
                        </v-col>
                    </v-row>
                </v-card-text>
            </v-card>
            <div v-if="loading" class="d-flex justify-center py-12">
                <v-progress-circular indeterminate color="primary" size="40" width="4" />
            </div>

            <!-- Error -->
            <v-alert v-else-if="error" type="error" variant="tonal" rounded="lg" class="mb-4">
                {{ error }}
            </v-alert>

            <!-- Table -->
            <v-card v-else variant="tonal" class="bg-surface-container" rounded="lg">
                <v-table density="comfortable" hover>
                    <thead>
                        <tr>
                            <th style="width: 40px">
                                <v-checkbox :model-value="isAllSelected"
                                    :indeterminate="selectedCount > 0 && !isAllSelected" density="compact" hide-details
                                    @click="toggleSelectAll" />
                            </th>
                            <th class="text-caption text-on-surface-variant">ID</th>
                            <th class="text-caption text-on-surface-variant">用户名</th>
                            <th class="text-caption text-on-surface-variant">角色</th>
                            <th class="text-caption text-on-surface-variant">IP</th>
                            <th class="text-caption text-on-surface-variant">日配额</th>
                            <th class="text-caption text-on-surface-variant">今日用量</th>
                            <th class="text-caption text-on-surface-variant text-right">操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="user in filteredUsers" :key="user.id"
                            :class="{ 'bg-surface-container-high': selectedIds.has(user.id) }">
                            <td>
                                <v-checkbox :model-value="selectedIds.has(user.id)" :disabled="user.is_admin"
                                    density="compact" hide-details @click="toggleSelect(user.id)" />
                            </td>
                            <td class="text-caption text-on-surface-variant">{{ user.id }}</td>
                            <td>
                                <div class="d-flex align-center ga-2">
                                    <v-icon v-if="user.is_admin" size="16" color="primary">mdi-shield-star</v-icon>
                                    <v-icon v-else-if="!user.is_active" size="16"
                                        color="error">mdi-account-cancel</v-icon>
                                    <span :class="{ 'text-on-surface-variant': !user.is_active }">
                                        {{ user.username }}
                                    </span>
                                </div>
                            </td>
                            <td>
                                <v-chip size="x-small" :color="roleChip(user).color">
                                    {{ roleChip(user).text }}
                                </v-chip>
                            </td>
                            <td>
                                <div class="text-caption">
                                    <div v-if="user.registration_ip" class="d-flex align-center ga-1">
                                        <v-icon size="12" color="on-surface-variant">mdi-account-plus</v-icon>
                                        <span>{{ user.registration_ip }}</span>
                                    </div>
                                    <div v-for="(ip, i) in user.last_login_ips" :key="i"
                                        class="d-flex align-center ga-1">
                                        <v-icon size="12" color="on-surface-variant">mdi-login</v-icon>
                                        <span class="text-on-surface-variant">{{ ip }}</span>
                                    </div>
                                    <span v-if="!user.registration_ip && user.last_login_ips.length === 0"
                                        class="text-on-surface-variant">—</span>
                                </div>
                            </td>
                            <td>
                                <v-chip size="x-small">
                                    {{ quotaLabel(user.daily_quota) }}
                                </v-chip>
                            </td>
                            <td>
                                <div class="d-flex align-center ga-2" style="min-width: 80px">
                                    <v-progress-linear :model-value="user.is_admin
                                        ? 0
                                        : Math.min(
                                            100,
                                            (user.used_today /
                                                Math.max(1, user.daily_quota ?? globalDefaultQuota)) *
                                            100,
                                        )
                                        " :color="user.used_today >= (user.daily_quota ?? globalDefaultQuota) &&
                                            !user.is_admin
                                            ? 'error'
                                            : 'primary'
                                            " height="4" rounded style="max-width: 50px; min-width: 24px" />
                                    <span class="text-caption">
                                        {{
                                            user.is_admin
                                                ? '—'
                                                : `${user.used_today}/${user.daily_quota ?? globalDefaultQuota}`
                                        }}
                                    </span>
                                </div>
                            </td>
                            <td class="text-right">
                                <div v-if="!user.is_admin" class="d-flex ga-1 justify-end">
                                    <v-btn size="x-small" variant="tonal" color="primary" @click="openEditUser(user)">
                                        编辑
                                    </v-btn>
                                    <v-btn size="x-small" variant="tonal" :color="user.is_active ? 'error' : 'success'"
                                        :loading="banSaving" @click="toggleBan(user)">
                                        {{ user.is_active ? '封禁' : '解禁' }}
                                    </v-btn>
                                </div>
                                <span v-else class="text-caption text-on-surface-variant">—</span>
                            </td>
                        </tr>
                    </tbody>
                </v-table>
                <v-card-text v-if="filteredUsers.length === 0" class="text-center py-8 text-on-surface-variant">
                    {{ allUsers.length === 0 ? '暂无用户' : '无匹配结果' }}
                </v-card-text>
            </v-card>

            <!-- Bulk Quota Dialog -->
            <v-dialog v-model="bulkQuotaDialog" max-width="400">
                <v-card rounded="lg">
                    <v-card-title class="text-body-1 font-weight-medium">
                        批量设置配额 — {{ selectedCount }} 个用户
                    </v-card-title>
                    <v-card-text>
                        <v-text-field v-model.number="bulkQuota" label="每日配额（0=全局默认，-1=无限）" type="number"
                            variant="outlined" density="compact" :min="-1" />
                    </v-card-text>
                    <v-card-actions>
                        <v-spacer />
                        <v-btn variant="text" @click="bulkQuotaDialog = false">取消</v-btn>
                        <v-btn variant="tonal" color="primary" :loading="bulkLoading" @click="doBulkQuota">
                            确认
                        </v-btn>
                    </v-card-actions>
                </v-card>
            </v-dialog>

            <!-- Edit User Dialog -->
            <v-dialog v-model="editDialog" max-width="420">
                <v-card rounded="lg">
                    <v-card-title class="text-body-1 font-weight-medium">
                        编辑用户 — {{ editUsername }}
                    </v-card-title>
                    <v-card-text>
                        <v-alert v-if="editError" type="error" variant="tonal" rounded="lg" class="mb-3"
                            density="compact">
                            {{ editError }}
                        </v-alert>

                        <!-- Quota -->
                        <div class="text-subtitle-2 text-on-surface-variant mb-1">每日配额</div>
                        <v-text-field v-model.number="editQuota" label="0=全局默认，-1=无限" type="number" variant="outlined"
                            density="compact" :min="-1" class="mb-4" />

                        <v-divider class="mb-4" />

                        <!-- Password (optional) -->
                        <div class="text-subtitle-2 text-on-surface-variant mb-1">
                            修改密码<span class="text-caption text-on-surface-variant ml-1">（留空则不修改）</span>
                        </div>
                        <v-text-field v-model="editNewPassword" label="新密码" type="password" variant="outlined"
                            density="compact" class="mb-2" />
                        <v-text-field v-model="editConfirmPassword" label="确认新密码" type="password" variant="outlined"
                            density="compact" />
                    </v-card-text>
                    <v-card-actions>
                        <v-spacer />
                        <v-btn variant="text" @click="closeEditUser">取消</v-btn>
                        <v-btn variant="tonal" color="primary" :loading="editLoading" @click="doSaveEditUser">
                            保存
                        </v-btn>
                    </v-card-actions>
                </v-card>
            </v-dialog>

        </div>
        <!-- / User Management -->

        <!-- ====== Review Tab ====== -->
        <div class="tab-panel" :class="{ 'tab-panel--active': activeTab === 'review' }">
            <div class="mt-2">
                <div v-if="reviewLoading" class="d-flex justify-center py-12">
                    <v-progress-circular indeterminate color="primary" size="40" width="4" />
                </div>
                <v-card v-else variant="tonal" class="bg-surface-container" rounded="xl">
                    <v-table density="comfortable" hover>
                        <thead>
                            <tr>
                                <th class="text-caption text-on-surface-variant">任务 ID</th>
                                <th class="text-caption text-on-surface-variant">URL</th>
                                <th class="text-caption text-on-surface-variant">提交时间</th>
                                <th class="text-caption text-on-surface-variant">状态</th>
                                <th class="text-caption text-on-surface-variant text-right">操作</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="task in reviewTasks" :key="task.task_id">
                                <td class="text-caption text-on-surface-variant font-monospace">
                                    {{ task.task_id.slice(0, 8) }}…
                                </td>
                                <td>
                                    <span class="text-caption text-truncate d-inline-block" style="max-width: 300px">
                                        {{ parseTaskUrl(task) }}
                                    </span>
                                </td>
                                <td class="text-caption text-on-surface-variant">
                                    {{ new Date(task.created_at).toLocaleString('zh-CN') }}
                                </td>
                                <td>
                                    <v-chip size="x-small" :color="reviewStatusChip(task).color">
                                        {{ reviewStatusChip(task).text }}
                                    </v-chip>
                                </td>
                                <td class="text-right">
                                    <div class="d-flex ga-1 justify-end">
                                        <v-btn size="x-small" variant="tonal" color="success" :loading="reviewActing"
                                            @click="doApprove(task)">
                                            通过
                                        </v-btn>
                                        <v-btn size="x-small" variant="tonal" color="error" :loading="reviewActing"
                                            @click="doReject(task)">
                                            拒绝
                                        </v-btn>
                                    </div>
                                </td>
                            </tr>
                        </tbody>
                    </v-table>
                    <v-card-text v-if="reviewTasks.length === 0" class="text-center py-8 text-on-surface-variant">
                        暂无待审核任务
                    </v-card-text>
                </v-card>
            </div>
        </div>
    </div>
</template>

<style scoped lang="scss">
.admin-page {
    position: relative;
}

.quota-input :deep(input) {
    text-align: center;
}

.tab-panel {
    position: absolute;
    width: 100%;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.25s ease, transform 0.25s ease;
    pointer-events: none;
    visibility: hidden;
}

.tab-panel--active {
    position: relative;
    opacity: 1;
    transform: translateY(0);
    pointer-events: auto;
    visibility: visible;
}
</style>
