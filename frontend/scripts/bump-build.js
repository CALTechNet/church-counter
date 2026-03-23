// Increments the BUILD constant in src/version.js before every build.
import { readFileSync, writeFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __dirname = dirname(fileURLToPath(import.meta.url))
const versionPath = join(__dirname, '../src/version.js')

const content = readFileSync(versionPath, 'utf8')
const match = content.match(/export const BUILD\s*=\s*(\d+)/)
if (!match) {
  console.error('ERROR: Could not find BUILD constant in version.js')
  process.exit(1)
}

const newBuild = parseInt(match[1], 10) + 1
const newContent = content.replace(
  /export const BUILD\s*=\s*\d+/,
  `export const BUILD   = ${newBuild}`
)
writeFileSync(versionPath, newContent)
console.log(`Build number bumped: ${match[1]} → ${newBuild}`)
