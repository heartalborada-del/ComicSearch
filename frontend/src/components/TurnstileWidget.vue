<script setup lang="ts">
/**
 * Turnstile widget component — renders a Cloudflare Turnstile challenge.
 * Emits a verified token on successful completion.
 *
 * Handles async siteKey resolution: the parent may provide siteKey after
 * mount (e.g. after an API call to /auth/status completes), so we watch
 * for siteKey changes and initialize the widget reactively.
 */
import { nextTick, onUnmounted, shallowRef, ref, watch } from 'vue'
import { loadTurnstileScript, useTurnstile, type TurnstileInstance } from '@/composables/useTurnstile'

const emit = defineEmits<{
  verify: [token: string]
  error: [message: string]
}>()

const props = withDefaults(
  defineProps<{
    siteKey: string
    resetKey?: number
  }>(),
  {
    resetKey: 0,
  },
)

const containerRef = ref<HTMLElement | null>(null)
const loading = ref(true)
const loadError = ref<string | null>(null)
const turnstile = shallowRef<TurnstileInstance | null>(null)

/**
 * Initialise (or re-initialise) the Turnstile widget.
 * Creates a fresh TurnstileInstance and renders it into the container.
 */
async function initTurnstile(): Promise<void> {
  if (!props.siteKey) return

  // Destroy previous instance if any
  turnstile.value?.destroy()
  turnstile.value = null

  loading.value = true
  loadError.value = null

  try {
    await loadTurnstileScript()
    // Must set loading=false BEFORE nextTick so the v-else-if="siteKey"
    // branch renders and containerRef binds to the actual DOM element.
    loading.value = false
    await nextTick()
    if (containerRef.value) {
      const instance = useTurnstile(props.siteKey)
      instance.render(containerRef.value)
      turnstile.value = instance
    }
  } catch (err) {
    loadError.value = '验证码加载失败'
    emit('error', '验证码加载失败')
    loading.value = false
  }
}

// Watch siteKey changes — the key may arrive asynchronously after mount
watch(
  () => props.siteKey,
  (newKey) => {
    if (newKey) {
      initTurnstile()
    } else {
      loading.value = false
    }
  },
  { immediate: true },
)

// Watch token changes and emit verify
watch(
  () => turnstile.value?.token.value,
  (newToken) => {
    if (newToken) {
      emit('verify', newToken)
    }
  },
)

// Watch for reset key changes
watch(
  () => props.resetKey,
  () => {
    turnstile.value?.reset()
  },
)

onUnmounted(() => {
  turnstile.value?.destroy()
})
</script>

<template>
  <div class="turnstile-widget">
    <!-- Loading state -->
    <div
      v-if="loading"
      class="d-flex align-center justify-center pa-4"
      style="min-height: 65px"
    >
      <v-progress-circular
        indeterminate
        size="20"
        width="2"
        color="primary"
        class="mr-2"
      />
      <span class="text-caption text-on-surface-variant">加载验证码…</span>
    </div>

    <!-- Error state -->
    <v-alert
      v-else-if="loadError"
      type="warning"
      variant="tonal"
      density="compact"
      rounded="lg"
      closable
    >
      {{ loadError }}
    </v-alert>

    <!-- Turnstile container -->
    <div
      v-else-if="siteKey"
      ref="containerRef"
      class="turnstile-container d-flex justify-center"
    />

    <!-- No site key configured -->
    <div v-else class="text-caption text-on-surface-variant text-center pa-2">
      验证码未配置
    </div>
  </div>
</template>

<style scoped lang="scss">
.turnstile-container {
  min-height: 65px;
}
</style>
