"""Die 7 Tavily-Rollen. Jeder Aufruf muss Frage beantworten, die Agent sonst nicht beantworten kann."""

from __future__ import annotations

from tavily import TavilyClient

from .config import Settings, load_settings

RESEARCH_DOMAINS = ["github.com", "arxiv.org", "openreview.net"]
ERROR_DOMAINS = ["stackoverflow.com", "github.com", "discuss.pytorch.org", "pytorch.org", "numpy.org"]


class TavilyTools:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._client = TavilyClient(api_key=self.settings.tavily_api_key)
        self._research_calls = 0  # max 1x pro Paper
        self.on_call = None       # Rueckruf (role, endpoint, target, result) fuers Ereignisprotokoll

    def _note(self, role: int, endpoint: str, target, result):
        if self.on_call:
            try:
                n = len((result or {}).get("results", [])) if isinstance(result, dict) else 0
            except (TypeError, AttributeError):
                n = 0
            self.on_call(role, endpoint, target, n)
        return result

    def find_material(self, paper_title: str, domains: list[str] | None = None) -> dict:
        """Rolle 1: Repo, Datensatz, Projektseite finden."""
        return self._note(1, "search", paper_title, self._client.search(
            query=paper_title,
            include_domains=domains or RESEARCH_DOMAINS,
        ))

    def read_claims(self, arxiv_html_url: str) -> dict:
        """Rolle 2: Tabellen mit Zielwerten aus arXiv-HTML extrahieren."""
        return self._note(2, "extract", arxiv_html_url,
                          self._client.extract(urls=[arxiv_html_url], extract_depth="advanced"))

    def collect_objections(self, paper_title: str, repo_url: str | None = None, limit: int = 5) -> dict:
        """Rolle 3: Issues, Gutachten, Kritik-Papers -> neue Einwaende.

        Search (mit Rohtext der Treffer) plus, falls das Repo bekannt ist, ein Crawl der Issue-Seite.
        `limit` begrenzt beides (Tavily-Credits). Rueckgabe: {"search": ..., "crawl": ... | None}.
        (Vorher: `crawl(urls=...)`, die API kennt nur `url`: der Aufruf waere mit TypeError gescheitert.)
        """
        search = self._client.search(
            query=f"{paper_title} reproducibility issues criticism evaluation", max_results=limit,
            include_raw_content="text",
        )
        crawl = None
        if repo_url:
            try:
                crawl = self._client.crawl(
                    url=repo_url.rstrip("/") + "/issues", limit=limit, max_depth=1, chunks_per_source=3,
                    instructions="Issues about the evaluation protocol, test set, drop_last, data leakage or "
                                 "whether the reported results can be reproduced",
                )
            except Exception as exc:   # Suchergebnis (schon bezahlt) nicht wegen eines Crawl-Fehlers verlieren
                crawl = {"error": f"{type(exc).__name__}: {exc}", "results": []}
        self._note(3, "search", paper_title, search)
        if repo_url:
            self._note(3, "crawl", repo_url.rstrip("/") + "/issues", crawl)
        return {"search": search, "crawl": crawl}

    def search_community(self, paper_title: str, limit: int = 5) -> dict:
        """Rolle 3 (Community-Check): Search nach Kritik und Reproduktionsberichten, mit Rohtext der Treffer. Nur Orte, an denen
        Diskussion steht; Zitier-Aggregatoren sind ausgeschlossen (die eigene arXiv-Seite filtert `community.drop_noise`)."""
        return self._note(3, "search", paper_title, self._client.search(
            query=f'{paper_title} reproduce OR reproducibility OR "could not reproduce" OR issue',
            max_results=limit, include_raw_content="text",
            include_domains=["github.com", "openreview.net", "arxiv.org", "reddit.com"],
            exclude_domains=["scirp.org", "semanticscholar.org", "researchgate.net", "scholar.google.com", "paperswithcode.com", "connectedpapers.com"],
        ))

    def search_method_critique(self, method: str, dataset: str, limit: int = 5) -> dict:
        """Rolle 3: methodenzentrierte Suche nach Kritik-Papers und Re-Evaluationen (z. B. Shchur 2018 "Pitfalls" fuer GCN auf Cora).
        Ergaenzt die Titel-Suche, ersetzt sie nicht."""
        return self._note(3, "search", f"{method} {dataset}", self._client.search(
            query=f'{method} {dataset} reproducibility OR re-evaluation OR pitfalls OR "could not reproduce"',
            max_results=limit, include_raw_content="text", include_domains=["arxiv.org", "openreview.net"],
        ))

    def search_repo_issues(self, owner_repo: str, limit: int = 5) -> dict:
        """Rolle 3: Issues des Paper-Repos per Suche statt der drei neuesten per Crawl: Query "<repo> reproduce results accuracy",
        beschraenkt auf github.com/<owner>/<repo>."""
        repo = owner_repo.split("/")[-1]
        return self._note(3, "search", owner_repo, self._client.search(
            query=f"{repo} reproduce results accuracy", max_results=limit, include_raw_content="text",
            include_domains=[f"github.com/{owner_repo}"],
        ))

    def crawl_issues(self, repo_url: str, limit: int = 3) -> dict:
        """Rolle 3: Issue-Seite des Repos (immer mit `limit`). Fehler werden als {"error", "results": []} zurueckgegeben."""
        url = repo_url.rstrip("/") + "/issues"
        try:
            res = self._client.crawl(
                url=url, limit=limit, max_depth=1, chunks_per_source=3,
                instructions="Issues about the evaluation protocol, test set, drop_last, data leakage or "
                             "whether the reported results can be reproduced",
            )
        except Exception as exc:   # das schon bezahlte Search-Ergebnis nicht wegen eines Crawl-Fehlers verlieren
            return {"error": f"{type(exc).__name__}: {exc}", "results": []}
        return self._note(3, "crawl", url, res)

    def read_pages(self, urls: list[str], limit: int = 3) -> dict:
        """Rolle 3: Seiten lesen, deren URLs schon bekannt sind (Discovery-Community-Treffer). Ein Extract statt einer neuen Suche."""
        return self._note(3, "extract", urls[:limit], self._client.extract(urls=urls[:limit], extract_depth="basic"))

    def get_setup_details(self, url: str) -> dict:
        """Rolle 4: Hyperparameter, Datenaufteilung aus README/Configs/Anhang."""
        return self._note(4, "extract", url, self._client.extract(urls=[url], extract_depth="advanced"))

    def resolve_error(self, error_text: str, max_results: int = 3) -> dict:
        """Rolle 5: Loesungshinweis fuer `code` bei Fehler (Suche mit der Fehlerzeile)."""
        return self._note(5, "search", error_text, self._client.search(
            query=error_text, max_results=max_results, include_domains=ERROR_DOMAINS,
        ))

    def clarify_discrepancy(self, paper_title: str, max_results: int = 3) -> dict:
        """Rolle 6: neuere arXiv-Version, Errata, Issues bei Abweichung."""
        return self._note(6, "search", paper_title, self._client.search(
            query=f"{paper_title} errata updated results erratum", max_results=max_results, include_raw_content="text"))

    def cite_sources(self, urls: list[str], paper_title: str) -> dict:
        """Rolle 7: zitierfaehige Quellen fuers Urteil. Research max 1x pro Paper."""
        extract_result = self._client.extract(urls=urls, extract_depth="advanced")
        if self._research_calls == 0:
            self._research_calls += 1
            research_result = self._client.search(query=paper_title, search_depth="advanced")
            return {"extract": extract_result, "research": research_result}
        return {"extract": extract_result}

    def curate_repo(self, paper_title: str) -> dict:
        """Discovery: Finde offizielle Github Implementierung."""
        query = f'"{paper_title}" github implementation'
        return self._note(8, "search", paper_title, self._client.search(
            query=query,
            include_domains=["github.com"],
            max_results=5,
        ))


    def curate_papers(self, query: str, max_results: int = 10) -> dict:
        """Discovery: aktuelle Papers mit Ergebnissen auf einem unterstuetzten Datensatz finden.

        Anders als die arXiv-API durchsucht Tavily auch OpenReview und liefert Textausschnitte mit Zahlen.
        """
        return self._note(8, "search", query, self._client.search(
            query=query,
            include_domains=["arxiv.org", "openreview.net"],
            time_range="year",
            search_depth="advanced",
            max_results=max_results,
        ))

    def curate_discussion(self, paper_title: str, max_results: int = 5) -> dict:
        """Discovery: Community-Hinweise zu einem Paper (Issues, Reviews, Diskussionen).

        Treffer werden nur als Belege angezeigt; ein Urteil daraus faellt erst im Loop (Rolle 3).
        """
        return self._note(8, "search", paper_title, self._client.search(
            query=f'"{paper_title}" reproduce OR reproducibility OR issue OR results',
            include_domains=["github.com", "openreview.net", "reddit.com"],
            max_results=max_results,
        ))

    def curate_extract(self, url: str, query: str, chunks_per_source: int = 5) -> dict:
        """Discovery: nur die zur Frage passenden Abschnitte eines Papers (Protokoll, Split, Ergebnisse).

        Tavily waehlt die Abschnitte; das spart LLM-Tokens und Wartezeit gegenueber dem Volltext.
        """
        return self._note(2, "extract", url, self._client.extract(
            urls=[url],
            extract_depth="advanced",
            format="markdown",
            query=query,
            chunks_per_source=chunks_per_source,
        ))
