<script setup lang="ts">
/**
 * Admin page — user management with search, bulk operations, quota editing, and IP records.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { getJson, postJson, ApiError } from '@/api/client'
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

// ---- State ----
const allUsers = ref<AdminUserItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)

// Search & filter
const searchQuery = ref('')
const filterStatus = ref<'all' | 'active' | 'banned'>('all')
const filterRole = ref<'all' | 'admin' | 'user'>('all')

// Selection
const selectedIds = ref<Set<number>>(new Set())

// Bulk operations
const bulkLoading = ref(false)
const bulkQuotaDialog = ref(false)
const bulkQuota = ref<number>(0)

// Quota editing (single)
const editingUserId = ref<number | null>(null)
const editingQuota = ref<number>(0)
const quotaSaving = ref(false)
const banSaving = ref(false)
const toast = ref<{ type: 'success' | 'error'; text: string } | null>(null)

const globalDefaultQuota = computed(() => authStore.quota?.daily_quota ?? 50)

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
function startEdit(user: AdminUserItem): void {
    editingUserId.value = user.id
    editingQuota.value = user.daily_quota ?? globalDefaultQuota.value
    toast.value = null
}

function cancelEdit(): void {
    editingUserId.value = null
    toast.value = null
}

async function saveQuota(user: AdminUserItem): Promise<void> {
    quotaSaving.value = true
    toast.value = null
    try {
        await postJson<{ count: number }>('/auth/quota/set', {
            user_ids: [user.id],
            daily_quota: editingQuota.value,
        })
        if (editingQuota.value <= 0) {
            user.daily_quota = null
        } else {
            user.daily_quota = editingQuota.value
        }
        editingUserId.value = null
        showToast('success', `${user.username} 配额已更新`)
    } catch (err) {
        showToast('error', err instanceof ApiError ? err.detail : '更新失败')
    } finally {
        quotaSaving.value = false
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

onMounted(loadUsers)
</script>

<template>
    <div class="admin-page">
        <!-- Header -->
        <div class="d-flex align-center ga-3 mb-4 flex-wrap">
            <h1 class="text-h5 font-weight-medium">用户管理</h1>
            <v-chip v-if="allUsers.length" size="small" variant="tonal" color="primary">
                {{ allUsers.length }} 个用户
            </v-chip>
            <v-chip v-if="selectedCount > 0" size="small" variant="tonal" color="secondary">
                已选 {{ selectedCount }}
            </v-chip>
        </div>

        <!-- Toast -->
        <v-alert v-if="toast" :type="toast.type" variant="tonal" rounded="lg" class="mb-3" closable density="compact">
            {{ toast.text }}
        </v-alert>

        <!-- Toolbar -->
        <v-card variant="tonal" class="bg-surface-container mb-4" rounded="xl">
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

        <!-- Loading -->
        <div v-if="loading" class="d-flex justify-center py-12">
            <v-progress-circular indeterminate color="primary" size="40" width="4" />
        </div>

        <!-- Error -->
        <v-alert v-else-if="error" type="error" variant="tonal" rounded="lg" class="mb-4">
            {{ error }}
        </v-alert>

        <!-- Table -->
        <v-card v-else variant="tonal" class="bg-surface-container" rounded="xl">
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
                                <v-icon v-else-if="!user.is_active" size="16" color="error">mdi-account-cancel</v-icon>
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
                                <div v-for="(ip, i) in user.last_login_ips" :key="i" class="d-flex align-center ga-1">
                                    <v-icon size="12" color="on-surface-variant">mdi-login</v-icon>
                                    <span class="text-on-surface-variant">{{ ip }}</span>
                                </div>
                                <span v-if="!user.registration_ip && user.last_login_ips.length === 0"
                                    class="text-on-surface-variant">—</span>
                            </div>
                        </td>
                        <td>
                            <template v-if="editingUserId === user.id">
                                <div class="d-flex align-center ga-1">
                                    <v-text-field v-model.number="editingQuota" type="number" density="compact"
                                        variant="outlined" hide-details class="quota-input" style="max-width: 80px"
                                        :min="-1" />
                                    <v-btn size="x-small" color="primary" variant="tonal" :loading="quotaSaving"
                                        @click="saveQuota(user)">
                                        <v-icon size="14">mdi-check</v-icon>
                                    </v-btn>
                                    <v-btn size="x-small" variant="text" @click="cancelEdit">
                                        <v-icon size="14">mdi-close</v-icon>
                                    </v-btn>
                                </div>
                            </template>
                            <v-chip v-else size="x-small">
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
                                <v-btn size="x-small" variant="tonal" color="primary"
                                    :disabled="editingUserId === user.id" @click="startEdit(user)">
                                    配额
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
            <v-card rounded="xl">
                <v-card-title class="text-body-1 font-weight-medium">
                    批量设置配额 — {{ selectedCount }} 个用户
                </v-card-title>
                <v-card-text>
                    <v-text-field v-model.number="bulkQuota" label="每日配额（0=全局默认，-1=无限）" type="number" variant="outlined"
                        density="compact" :min="-1" />
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
    </div>
</template>

<style scoped lang="scss">
.quota-input :deep(input) {
    text-align: center;
}
</style>
