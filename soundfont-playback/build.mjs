import {build} from 'esbuild';
import {copyFile, mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';

const source = new URL('./', import.meta.url);
const publishFrontend = process.argv.includes('--frontend');
const target = new URL(publishFrontend ? '../frontend/vendor/soundfont/' : './', source);
const dist = new URL('dist/', target);
const local = (file, dir = source) => fileURLToPath(new URL(file, dir));

await mkdir(dist, {recursive:true});
await build({
  entryPoints:[local('src/player.mjs')],
  outfile:local('grouped-player.mjs', dist),
  bundle:true,
  format:'esm',
  minify:true,
});
await copyFile(local('node_modules/spessasynth_lib/dist/spessasynth_processor.min.js'),
  local('spessasynth_processor.min.js', dist));

if (publishFrontend) {
  await copyFile(local('soundfonts.json'), local('soundfonts.json', target));
  const notices = new URL('licenses/', target);
  await mkdir(notices, {recursive:true});
  for (const dep of ['spessasynth_core', 'spessasynth_lib', 'stb-vorbis', 'fflate']) {
    await copyFile(local(`node_modules/${dep}/LICENSE`), local(`${dep}.txt`, notices));
  }
  await copyFile(local('LICENSE-MuScriptor.txt'), local('MuScriptor.txt', notices));
}
