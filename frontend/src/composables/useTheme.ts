/**
 * Theme switching composable.
 * Manages light/dark theme toggle with localStorage persistence.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useTheme } from 'vuetify'

const STORAGE_KEY = 'comicsearch-theme'

const isDark = ref(false)

export function useAppTheme() {
    const vuetifyTheme = useTheme()

    const themeName = computed(() => (isDark.value ? 'm3Dark' : 'm3Light'))

    function applyTheme() {
        vuetifyTheme.global.name.value = themeName.value
    }
    function toggleTheme() {
        isDark.value = !isDark.value
    }

    function setTheme(dark: boolean) {
        isDark.value = dark
    }

    function loadFromStorage() {
        const stored = localStorage.getItem(STORAGE_KEY)
        if (stored === 'dark') {
            isDark.value = true
        } else if (stored === 'light') {
            isDark.value = false
        } else {
            // Follow system preference on first visit
            isDark.value = window.matchMedia('(prefers-color-scheme: dark)').matches
        }
        applyTheme()
    }

    watch(isDark, () => {
        localStorage.setItem(STORAGE_KEY, isDark.value ? 'dark' : 'light')
        applyTheme()
    })

    onMounted(() => {
        loadFromStorage()
    })

    return {
        isDark,
        themeName,
        toggleTheme,
        setTheme,
    }
}
