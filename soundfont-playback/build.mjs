import {build} from 'esbuild';
import {copyFile, mkdir} from 'node:fs/promises';
await mkdir('dist',{recursive:true});
await build({entryPoints:['src/player.mjs'],outfile:'dist/grouped-player.mjs',bundle:true,format:'esm',minify:true});
await copyFile('node_modules/spessasynth_lib/dist/spessasynth_processor.min.js','dist/spessasynth_processor.min.js');
