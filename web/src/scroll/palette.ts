// Palettes. "paper" carries words (claims, sentences); "night" carries the machinery (sandbox, harness, benchmark).
export type Theme = 'paper' | 'night'
export const PAL = {
  paper: { bg: '#f4f0e8', card: '#fbf9f4', ink: '#151310', ink2: '#4a463f', muted: '#8a857b', line: '#ddd6c9', faint: '#e9e3d8', green: '#047857', greenSoft: 'rgba(4,120,87,0.12)', red: '#be123c', redSoft: 'rgba(190,18,60,0.10)', amber: '#b45309' },
  night: { bg: '#0b0a09', card: '#14130f', ink: '#f1eee6', ink2: '#b9b5aa', muted: '#7d796f', line: '#2a2823', faint: '#1b1a16', green: '#34d399', greenSoft: 'rgba(52,211,153,0.14)', red: '#fb7185', redSoft: 'rgba(251,113,133,0.12)', amber: '#fbbf24' },
}
export type Pal = typeof PAL.paper

