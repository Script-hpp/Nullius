// Human label for a run header: "disc_2307_03756" and "disc_2410_09348_d8991277dd33d43f" -> "arXiv 2307.03756"; other ids stay readable ("kipf_gcn_2017" -> "kipf gcn 2017").
export function paperLabel(id: string): string {
  const m = id.match(/^disc_(\d{4})_(\d{4,5})(?:_[0-9a-f]+)?$/)
  return m ? `arXiv ${m[1]}.${m[2]}` : id.replace(/_/g, ' ')
}
