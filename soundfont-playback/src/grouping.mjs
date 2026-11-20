// Instrument classification adapted from MuScriptor MT3_FULL_PLUS.
// Copyright (c) 2026 Kyutai x Mirelo, MIT. See LICENSE-MuScriptor.txt.
const rows = [
  [0, 'Acoustic piano', 'piano', [0, 1, 3, 6, 7]],
  [1, 'Electric piano', 'keys', [2, 4, 5]],
  [2, 'Chromatic percussion', 'keys', [8, 9, 10, 11, 12, 13, 14, 15]],
  [3, 'Organ', 'keys', [16, 17, 18, 19, 20, 21, 22, 23]],
  [4, 'Acoustic guitar', 'guitars', [24, 25]],
  [5, 'Clean electric guitar', 'guitars', [26, 27, 28]],
  [6, 'Distorted electric guitar', 'guitars', [29, 30, 31]],
  [7, 'Acoustic bass', 'guitars', [32, 35]],
  [8, 'Electric bass', 'guitars', [33, 34, 36, 37, 38, 39]],
  [9, 'Violin', 'strings', [40]],
  [10, 'Viola', 'strings', [41]],
  [11, 'Cello', 'strings', [42]],
  [12, 'Contrabass', 'strings', [43]],
  [13, 'Orchestral harp', 'strings', [46]],
  [14, 'Timpani', 'strings', [47]],
  [15, 'String ensemble', 'strings', [48, 49, 44, 45]],
  [16, 'Synth strings', 'strings', [50, 51]],
  [17, 'Voice', 'strings', [52, 53, 54]],
  // Group 18 (GM 55, orchestra hit) is intentionally unsupported.
  [19, 'Trumpet', 'winds', [56, 59]],
  [20, 'Trombone', 'winds', [57]],
  [21, 'Tuba', 'winds', [58]],
  [22, 'French horn', 'winds', [60]],
  [23, 'Brass section', 'winds', [61, 62, 63]],
  [24, 'Soprano and alto sax', 'winds', [64, 65]],
  [25, 'Tenor sax', 'winds', [66]],
  [26, 'Baritone sax', 'winds', [67]],
  [27, 'Oboe', 'winds', [68]],
  [28, 'English horn', 'winds', [69]],
  [29, 'Bassoon', 'winds', [70]],
  [30, 'Clarinet', 'winds', [71]],
  [31, 'Flutes', 'winds', [72, 73, 74, 75, 76, 77, 78, 79]],
  [32, 'Synth lead', 'synth', [80, 81, 82, 83, 84, 85, 86, 87]],
  [33, 'Synth pad', 'synth', [88, 89, 90, 91, 92, 93, 94, 95]],
  [34, 'Atmosphere', 'synth', [100]],
  [35, 'Brightness', 'synth', [101]],
  [36, 'Drums', 'drums', [128]],
];
export const INSTRUMENT_GROUPS = Object.freeze(rows.map(([id, label, bank, programs]) =>
  Object.freeze({id, label, bank, program: programs[0], programs: Object.freeze(programs)})));
const byProgram = new Map(INSTRUMENT_GROUPS.flatMap(group => group.programs.map(program => [program, group])));
const validNote = n => n && Number.isInteger(n.pitch) && n.pitch >= 0 && n.pitch < 128 &&
  Number.isFinite(n.onset) && n.onset >= 0 && Number.isFinite(n.offset) && n.offset > n.onset;

// Groups are a playback view. Neither original note objects nor exported MIDI are rewritten.
export function groupPlaybackNotes(notes) {
  const groups = new Map();
  const unsupported = new Set();
  let invalidNoteCount = 0;
  for (const n of notes) {
    if (!validNote(n)) { invalidNoteCount++; continue; }
    const program = n.isDrum ? 128 : n.program;
    const group = byProgram.get(program);
    if (!group) { unsupported.add(program); continue; }
    if (!groups.has(group.id)) groups.set(group.id, {...group, notes: []});
    groups.get(group.id).notes.push(n);
  }
  const tracks = [...groups.values()].sort((a, b) => a.id - b.id);
  return {tracks, banks: [...new Set(tracks.map(track => track.bank))],
    unsupportedPrograms: [...unsupported], invalidNoteCount};
}
