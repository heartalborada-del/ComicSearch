/**
 * Vuetify plugin configuration with Material You (M3) themes,
 * typography scale, and default component variants.
 */
import 'vuetify/styles'
import '@mdi/font/css/materialdesignicons.css'
import { createVuetify, type ThemeDefinition } from 'vuetify'

import { lightTheme, darkTheme } from '@/styles/theme'

const m3Light: ThemeDefinition = {
    dark: false,
    colors: lightTheme.colors,
    variables: lightTheme.variables,
}

const m3Dark: ThemeDefinition = {
    dark: true,
    colors: darkTheme.colors,
    variables: darkTheme.variables,
}

export default createVuetify({
    theme: {
        defaultTheme: 'm3Light',
        themes: {
            m3Light,
            m3Dark,
        },
    },
    defaults: {
        VBtn: {
            variant: 'tonal',
            rounded: 'lg',
        },
        VCard: {
            rounded: 'lg',
            elevation: 0,
        },
        VChip: {
            variant: 'tonal',
        },
        VTextField: {
            variant: 'outlined',
            density: 'comfortable',
        },
        VTextarea: {
            variant: 'outlined',
            density: 'comfortable',
        },
        VSelect: {
            variant: 'outlined',
            density: 'comfortable',
        },
        VCombobox: {
            variant: 'outlined',
            density: 'comfortable',
        },
        VSwitch: {
            color: 'primary',
            inset: true,
        },
        VSlider: {
            color: 'primary',
            showTicks: 'always',
            thumbLabel: true,
        },
        VList: {
            rounded: 'lg',
        },
        VNavigationDrawer: {
            rounded: '0',
        },
    },
    display: {
        mobileBreakpoint: 'md',
        thresholds: {
            xs: 0,
            sm: 600,
            md: 960,
            lg: 1280,
            xl: 1920,
            xxl: 2560,
        },
    },
})
