<script setup lang="ts">
/**
 * Task item card — displays a single task with status, timestamps,
 * cancel button, and expandable error details.
 */
import { computed } from 'vue'
import type { TaskRecord, TaskStatus } from '@/types/task'

const props = defineProps<{
  task: TaskRecord
}>()

const emit = defineEmits<{
  cancel: [taskId: string]
}>()

const statusConfig: Record<TaskStatus, { color: string; icon: string; label: string }> = {
  pending_review: { color: 'warning', icon: 'mdi-file-document-edit', label: '待审核' },
  pending: { color: 'grey', icon: 'mdi-clock-outline', label: '等待中' },
  running: { color: 'primary', icon: 'mdi-progress-clock', label: '运行中' },
  success: { color: 'success', icon: 'mdi-check-circle', label: '成功' },
  failed: { color: 'error', icon: 'mdi-alert-circle', label: '失败' },
  cancelled: { color: '', icon: 'mdi-cancel', label: '已取消' },
}

const config = computed(() => statusConfig[props.task.status] ?? statusConfig.pending)

const canCancel = computed(
  () =>
    (props.task.status === 'pending' || props.task.status === 'running') &&
    !props.task.cancel_requested,
)

const shortId = computed(() => props.task.task_id.slice(0, 8))

const formattedCreatedAt = computed(() => formatTime(props.task.created_at))
const formattedStartedAt = computed(() => formatTime(props.task.started_at))
const formattedFinishedAt = computed(() => formatTime(props.task.finished_at))

function formatTime(iso: string | null): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return iso
  }
}

function onCancel(): void {
  emit('cancel', props.task.task_id)
}
</script>

<template>
  <v-card variant="tonal" class="bg-surface-container task-card h-100" rounded="lg">
    <v-card-text class="pa-4">
      <!-- Header: ID + Status -->
      <div class="d-flex justify-space-between align-start mb-3">
        <div>
          <div class="text-body-1 font-weight-medium">
            {{ task.task_type }}
          </div>
          <div class="text-caption text-on-surface-variant">
            ID: {{ shortId }}…
          </div>
        </div>
        <div class="d-flex flex-column align-end ga-1">
          <v-chip :color="config.color" size="small" :prepend-icon="config.icon">
            {{ config.label }}
          </v-chip>
          <v-chip
            v-if="task.cancel_requested"
            size="x-small"
            variant="tonal"
            color="warning"
          >
            取消请求中
          </v-chip>
        </div>
      </div>

      <!-- Timestamps -->
      <div class="text-caption text-on-surface-variant mb-2">
        <div>创建: {{ formattedCreatedAt }}</div>
        <div>开始: {{ formattedStartedAt }}</div>
        <div>完成: {{ formattedFinishedAt }}</div>
      </div>

      <!-- Error (expandable) -->
      <v-expansion-panels v-if="task.error" variant="accordion" class="mt-2">
        <v-expansion-panel>
          <v-expansion-panel-title class="text-error text-body-2">
            <v-icon start size="16">mdi-alert</v-icon>
            错误详情
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <pre class="text-caption text-error">{{ task.error }}</pre>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>

      <!-- Result (if success) -->
      <div v-if="task.result && task.status === 'success'" class="mt-2">
        <v-chip size="x-small" variant="tonal" color="success" prepend-icon="mdi-package-variant">
          Pack ID: {{ (task.result as Record<string, unknown>).pack_id ?? '—' }}
        </v-chip>
      </div>

      <!-- Actions -->
      <div v-if="canCancel" class="mt-3">
        <v-btn
          size="small"
          variant="tonal"
          color="error"
          prepend-icon="mdi-cancel"
          @click="onCancel"
        >
          取消任务
        </v-btn>
      </div>
    </v-card-text>
  </v-card>
</template>
