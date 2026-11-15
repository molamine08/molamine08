// Bundles src/ into one self-contained dist/index.html.
import { build } from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

const out = await build({
  entryPoints: ['src/app.js'],
  bundle: true,
  format: 'esm',
  target: ['safari15', 'chrome100'],
  charset: 'utf8',
  write: false,
  logLevel: 'warning',
});
const js = out.outputFiles[0].text;
if (js.includes('</script')) throw new Error('bundle contains a closing script tag');
const css = readFileSync('src/styles.css', 'utf8');
const html = readFileSync('src/index.html', 'utf8')
  .replace('/*__CSS__*/', () => css)
  .replace('/*__JS__*/', () => js);
mkdirSync('dist', { recursive: true });
writeFileSync('dist/index.html', html);
console.log(`dist/index.html  ${(html.length / 1024).toFixed(1)} KB`);
