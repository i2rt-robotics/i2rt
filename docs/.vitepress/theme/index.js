import DefaultTheme from 'vitepress/theme'
import './style.css'
import RobotViewer from './components/RobotViewer.vue'
import RobotCompare from './components/RobotCompare.vue'
import MediaPlaceholder from './components/MediaPlaceholder.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('RobotViewer', RobotViewer)
    app.component('RobotCompare', RobotCompare)
    app.component('MediaPlaceholder', MediaPlaceholder)
  },
}
