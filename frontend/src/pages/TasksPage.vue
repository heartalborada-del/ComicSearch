<script setup lang="ts">
/**
 * Tasks page — list, filter, and cancel async tasks.
 * Auto-polls for pending/running task updates.
 */
import { onMounted, onUnmounted } from 'vue'
import { useTasksStore } from '@/stores/tasks'
import type { TaskStatus } from '@/types/task'
import TaskItemCard from '@/components/TaskItemCard.vue'

const tasksStore = useTasksStore()

interface FilterOption {
  value: TaskStatus | undefined
  label: string
}

const filterOptions: FilterOption[] = [
  { value: undefined, label: '全部' },
  { value: 'pending', label: '等待中' },
  { value: 'running', label: '运行中' },
  { value: 'success', label: '成功' },
  { value: 'failed', label: '失败' },
]

function setFilter(status: TaskStatus | undefined): void {
  tasksStore.setStatusFilter(status)
}

async function handleCancel(taskId: string): Promise<void> {
  await tasksStore.cancelTaskById(taskId)
}

onMounted(async () => {
  await tasksStore.fetchTasks()
  if (tasksStore.hasActiveTasks) {
    tasksStore.startPolling()
  }
})

onUnmounted(() => {
  tasksStore.stopPolling()
})
</script>

<template>
  <div class="tasks-page">
    <div class="d-flex align-center justify-space-between mb-4 flex-wrap ga-2">
      <h1 class="text-h5 font-weight-medium">任务管理</h1>
      <v-btn
        variant="tonal"
        size="small"
        prepend-icon="mdi-refresh"
        :loading="tasksStore.loading"
        @click="tasksStore.fetchTasks()"
      >
        刷新
      </v-btn>
    </div>

    <!-- Status Filter Chips -->
    <div class="d-flex flex-wrap ga-2 mb-4">
      <v-chip
        v-for="option in filterOptions"
        :key="option.label"
        :variant="tasksStore.statusFilter === option.value ? 'flat' : 'tonal'"
        :color="tasksStore.statusFilter === option.value ? 'primary' : undefined"
        size="small"
        @click="setFilter(option.value)"
      >
        {{ option.label }}
        <v-badge
          v-if="option.value === undefined"
          :content="tasksStore.statusCounts.all"
          inline
          color="primary"
        />
        <template v-else>
          ({{ tasksStore.statusCounts[option.value] || 0 }})
        </template>
      </v-chip>
    </div>

    <!-- Loading -->
    <div v-if="tasksStore.loading && tasksStore.tasks.length === 0" class="d-flex justify-center py-12">
      <v-progress-circular indeterminate color="primary" size="48" width="4" />
    </div>

    <!-- Error -->
    <v-alert
      v-if="tasksStore.error"
      type="error"
      variant="tonal"
      rounded="lg"
      class="mb-4"
      closable
    >
      {{ tasksStore.error }}
    </v-alert>

    <!-- Task List -->
    <v-row v-if="tasksStore.filteredTasks.length > 0">
      <v-col
        v-for="task in tasksStore.filteredTasks"
        :key="task.task_id"
        cols="12"
        sm="6"
        md="4"
        lg="3"
      >
        <TaskItemCard :task="task" @cancel="handleCancel" />
      </v-col>
    </v-row>

    <!-- Empty State -->
    <div
      v-else-if="!tasksStore.loading"
      class="d-flex flex-column align-center justify-center py-12 text-center"
    >
      <v-icon size="80" color="outline" class="mb-4">mdi-format-list-checks</v-icon>
      <div class="text-h6 text-on-surface-variant mb-1">暂无任务</div>
      <div class="text-body-2 text-on-surface-variant mb-4">
        前往导入页面提交 E-Hentai 导入任务
      </div>
      <v-btn variant="tonal" color="primary" to="/import" prepend-icon="mdi-plus">
        新建导入
      </v-btn>
    </div>

    <!-- Polling indicator -->
    <div v-if="tasksStore.polling" class="text-caption text-on-surface-variant text-center mt-4">
      <v-progress-circular indeterminate size="12" width="2" color="primary" class="mr-1" />
      自动刷新中…
    </div>
  </div>
</template>
