<script setup lang="ts">
/**
 * Confidence badge — displays search confidence level as a colored chip.
 * - high: success/green
 * - medium: warning/amber
 * - low: error/red
 */
import { computed } from 'vue'
import type { ConfidenceLevel } from '@/types/search'

const props = defineProps<{
  confidence: ConfidenceLevel
}>()

const config = computed(() => {
  switch (props.confidence) {
    case 'high':
      return { color: 'success', icon: 'mdi-check-circle', label: '高置信度' }
    case 'medium':
      return { color: 'warning', icon: 'mdi-alert-circle', label: '中置信度' }
    case 'low':
      return { color: 'error', icon: 'mdi-alert-octagon', label: '低置信度' }
    default:
      return { color: 'grey', icon: 'mdi-help-circle', label: '未知' }
  }
})
</script>

<template>
  <v-chip :color="config.color" variant="tonal" size="small" :prepend-icon="config.icon">
    {{ config.label }}
  </v-chip>
</template>
