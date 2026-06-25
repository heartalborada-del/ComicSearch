<script setup lang="ts">
/**
 * Admin page — user management: list users, view usage, set per-user quota.
 */
import { computed, onMounted, ref } from 'vue'
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
}

interface SetQuotaResult {
    user_id: number
    username: string
    daily_quota: number | null
}

const authStore = useAuthStore()

// ---- State ----
const users = ref<AdminUserItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)

// Quota editing
const editingUserId = ref<number | null>(null)
const editingQuota = ref<number>(0)
const quotaSaving = ref(false)
const quotaMessage = ref<{ type: 'success' | 'error'; text: string } | null>(null)

const globalDefaultQuota = computed(() =>
    authStore.quota?.daily_quota ?? 50,
)

// ---- Actions ----
async function loadUsers(): Promise<void> {
    loading.value = true
    error.value = null
    try {
        users.value = await getJson<AdminUserItem[]>('/auth/users')
    } catch (err) {
        error.value = err instanceof ApiError ? err.detail : '加载用户列表失败'
    } finally {
        loading.value = false
    }
}

function startEdit(user: AdminUserItem): void {
    editingUserId.value = user.id
    editingQuota.value = user.daily_quota ?? globalDefaultQuota.value
    quotaMessage.value = null
}

function cancelEdit(): void {
    editingUserId.value = null
    quotaMessage.value = null
}

async function saveQuota(user: AdminUserItem): Promise<void> {
    quotaSaving.value = true
    quotaMessage.value = null
    try {
        const result = await postJson<SetQuotaResult>('/auth/quota/set', {
            user_id: user.id,
            daily_quota: editingQuota.value,
        })
        // Update local state
        user.daily_quota = result.daily_quota
        editingUserId.value = null
        quotaMessage.value = {
            type: 'success',
            text: `${result.username} 的配额已更新`,
        }
        setTimeout(() => { quotaMessage.value = null }, 3000)
    } catch (err) {
        quotaMessage.value = {
            type: 'error',
            text: err instanceof ApiError ? err.detail : '更新失败',
        }
    } finally {
        quotaSaving.value = false
    }
}

function quotaLabel(quota: number | null): string {
    if (quota === null || quota === 0) return `全局 (${globalDefaultQuota.value})`
    if (quota < 0) return '无限'
    return String(quota)
}

function quotaChipColor(user: AdminUserItem): string | undefined {
    if (user.is_admin) return 'primary'
    if (!user.is_active) return undefined
    const q = user.daily_quota ?? globalDefaultQuota.value
    if (q <= 0 || user.used_today < q) return undefined
    return 'error'
}

onMounted(loadUsers)
</script>

<template>
    <div class="admin-page">
        <div class="d-flex align-center ga-3 mb-4 flex-wrap">
            <h1 class="text-h5 font-weight-medium">用户管理</h1>
            <v-chip v-if="users.length" size="small" variant="tonal" color="primary">
                {{ users.length }} 个用户
            </v-chip>
        </div>

        <!-- Global message -->
        <v-alert v-if="quotaMessage" :type="quotaMessage.type" variant="tonal" rounded="lg" class="mb-4" closable
            density="compact">
            {{ quotaMessage.text }}
        </v-alert>

        <!-- Loading -->
        <div v-if="loading" class="d-flex justify-center py-12">
            <v-progress-circular indeterminate color="primary" size="40" width="4" />
        </div>

        <!-- Error -->
        <v-alert v-else-if="error" type="error" variant="tonal" rounded="lg" class="mb-4">
            {{ error }}
        </v-alert>

        <!-- User Table -->
        <v-card v-else variant="tonal" class="bg-surface-container" rounded="xl">
            <v-table density="comfortable" hover>
                <thead>
                    <tr>
                        <th class="text-caption text-on-surface-variant">ID</th>
                        <th class="text-caption text-on-surface-variant">用户名</th>
                        <th class="text-caption text-on-surface-variant">角色</th>
                        <th class="text-caption text-on-surface-variant">日配额</th>
                        <th class="text-caption text-on-surface-variant">今日用量</th>
                        <th class="text-caption text-on-surface-variant text-right">操作</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="user in users" :key="user.id">
                        <!-- ID -->
                        <td class="text-caption text-on-surface-variant">{{ user.id }}</td>

                        <!-- Username -->
                        <td>
                            <div class="d-flex align-center ga-2">
                                <v-icon v-if="user.is_admin" size="16" color="primary">mdi-shield-star</v-icon>
                                <v-icon v-else-if="!user.is_active" size="16" color="on-surface-variant">
                                    mdi-account-cancel
                                </v-icon>
                                <span :class="{ 'text-on-surface-variant': !user.is_active }">
                                    {{ user.username }}
                                </span>
                            </div>
                        </td>

                        <!-- Role -->
                        <td>
                            <v-chip size="x-small" :color="user.is_admin ? 'primary' : undefined">
                                {{ user.is_admin ? '管理员' : '用户' }}
                            </v-chip>
                        </td>

                        <!-- Daily Quota -->
                        <td>
                            <template v-if="editingUserId === user.id">
                                <div class="d-flex align-center ga-2">
                                    <v-text-field v-model.number="editingQuota" type="number" density="compact"
                                        variant="outlined" hide-details class="quota-input" style="max-width: 100px"
                                        :min="-1" />
                                    <v-btn size="x-small" color="primary" variant="tonal" :loading="quotaSaving"
                                        @click="saveQuota(user)">
                                        保存
                                    </v-btn>
                                    <v-btn size="x-small" variant="text" @click="cancelEdit">
                                        取消
                                    </v-btn>
                                </div>
                            </template>
                            <template v-else>
                                <v-chip size="x-small" :color="quotaChipColor(user)">
                                    {{ quotaLabel(user.daily_quota) }}
                                </v-chip>
                            </template>
                        </td>

                        <!-- Used Today -->
                        <td>
                            <div class="d-flex align-center ga-2">
                                <v-progress-linear :model-value="user.is_admin
                                        ? 0
                                        : Math.min(100, (user.used_today / Math.max(1, user.daily_quota ?? globalDefaultQuota)) * 100)
                                    " :color="user.used_today >= (user.daily_quota ?? globalDefaultQuota) && !user.is_admin ? 'error' : 'primary'"
                                    height="4" rounded style="max-width: 60px; min-width: 30px" />
                                <span class="text-caption">
                                    {{ user.is_admin ? '—' : `${user.used_today}/${user.daily_quota ??
                                    globalDefaultQuota}` }}
                                </span>
                            </div>
                        </td>

                        <!-- Actions -->
                        <td class="text-right">
                            <v-btn v-if="!user.is_admin" size="x-small" variant="tonal" color="primary"
                                :disabled="editingUserId === user.id" @click="startEdit(user)">
                                设置配额
                            </v-btn>
                            <span v-else class="text-caption text-on-surface-variant">—</span>
                        </td>
                    </tr>
                </tbody>
            </v-table>

            <v-card-text v-if="users.length === 0" class="text-center py-8 text-on-surface-variant">
                暂无用户
            </v-card-text>
        </v-card>
    </div>
</template>

<style scoped lang="scss">
.quota-input :deep(input) {
    text-align: center;
}
</style>
