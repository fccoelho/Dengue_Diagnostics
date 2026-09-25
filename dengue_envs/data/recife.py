"""Notificações de dengue e chikungunya do Recife: leitura, classificação e geocodificação.

Fonte: Portal de Dados Abertos da Prefeitura do Recife, conjunto "Casos de
Dengue, Zika e Chikungunya" (licença ODbL): uma linha por notificação, com
classificação final, critério de confirmação, laudos e endereço de residência
(rua e bairro), mas SEM coordenadas. A geocodificação usa duas bases do mesmo
portal, conjunto "Área urbana": os polígonos oficiais dos bairros e os trechos
de logradouros (geometria de cada rua, com o nome).

Os arquivos brutos ficam em `results/recife/` (fora do git): têm endereço.

Dois formatos convivem no portal:
- até 2020, nomes longos em minúsculas (`no_bairro_residencia`, ...);
- de 2021 em diante, os nomes do SINAN (`NM_BAIRRO`, `CLASSI_FIN`, ...).

Classificação final (SINAN): 1-4 e 10-12 dengue confirmada, 13 chikungunya,
5 descartado, 8 inconclusivo. Nos arquivos de chikungunya de 2015-16 aparecem
também os códigos 1 e 2 (escala antiga da dengue) sem nenhuma sorologia nem o
campo clínico de chik preenchidos: o significado é ambíguo, e esses registros
ficam de fora como `ambiguo`.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

RAIZ = Path("results/recife")
CODIGO_RECIFE = "261160"
# SIRGAS 2000 / UTM 25S: o fuso do Recife (o Rio usa o 23S).
CRS_RECIFE = "EPSG:31985"

# Formato SINAN (2021+) -> formato longo (até 2020).
_COLUNAS_SINAN = {
    "DT_NOTIFIC": "dt_notificacao",
    "DT_SIN_PRI": "dt_diagnostico_sintoma",
    "ID_MN_RESI": "co_municipio_residencia",
    "ID_DISTRIT": "co_distrito_residencia",
    "ID_BAIRRO": "co_bairro_residencia",
    "NM_BAIRRO": "no_bairro_residencia",
    "NM_LOGRADO": "nome_logradouro_residencia",
    "NU_CEP": "nu_cep_residencia",
    "CLASSI_FIN": "tp_classificacao_final",
    "CRITERIO": "tp_criterio_confirmacao",
}
_COLUNAS = list(_COLUNAS_SINAN.values())

DENGUE_CONFIRMADA = {1, 2, 3, 4, 10, 11, 12}


# --------------------------------------------------------------------------
# leitura
# --------------------------------------------------------------------------

def _le_csv(caminho: Path) -> pd.DataFrame:
    amostra = caminho.read_bytes()[:5000].decode("utf-8", "ignore").splitlines()[0]
    sep = ";" if amostra.count(";") > amostra.count(",") else ","
    return pd.read_csv(caminho, sep=sep, dtype=str, low_memory=False, encoding="utf-8")


def _inteiro(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def classifica(doenca_notificada: str, classificacao: pd.Series) -> pd.Series:
    """Status final da notificação: dengue, chik, descartado, inconclusivo ou ambiguo."""
    c = _inteiro(classificacao)
    status = pd.Series("sem_classificacao", index=classificacao.index, dtype=object)
    status[c == 5] = "descartado"
    status[c == 8] = "inconclusivo"
    status[c == 13] = "chik"
    dengue = c.isin(DENGUE_CONFIRMADA).fillna(False)
    if doenca_notificada == "dengue":
        status[dengue] = "dengue"
    else:
        # Ficha de chik com código da escala antiga da dengue: ambíguo (ver cabeçalho).
        status[dengue] = "ambiguo"
    return status


def carrega_notificacoes(raw: Path = RAIZ / "raw") -> pd.DataFrame:
    """Uma linha por notificação, colunas padronizadas, só residentes do Recife."""
    partes = []
    for caminho in sorted(raw.glob("*_20*.csv")):
        doenca, ano = caminho.stem.split("_")
        d = _le_csv(caminho)
        # O formato SINAN aparece em maiúsculas ou minúsculas, conforme o ano.
        d.columns = [c.strip() for c in d.columns]
        d = d.rename(columns=lambda c: _COLUNAS_SINAN.get(c.upper(), c).lower())
        faltam = [c for c in _COLUNAS if c not in d.columns]
        if faltam:
            raise KeyError(f"{caminho.name}: colunas ausentes {faltam}")
        d = d[_COLUNAS].copy()
        d["doenca_notificada"] = doenca
        d["ano_arquivo"] = int(ano)
        d["status"] = classifica(doenca, d["tp_classificacao_final"])
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    for c in ("dt_notificacao", "dt_diagnostico_sintoma"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["municipio"] = df["co_municipio_residencia"].str.extract(r"(\d{6})", expand=False)
    return df[df["municipio"] == CODIGO_RECIFE].reset_index(drop=True)


# --------------------------------------------------------------------------
# geocodificação
# --------------------------------------------------------------------------

_TIPOS = {
    "R": "RUA", "RUA": "RUA", "AV": "AVENIDA", "AVENIDA": "AVENIDA", "AVE": "AVENIDA",
    "TV": "TRAVESSA", "TRV": "TRAVESSA", "TRAV": "TRAVESSA", "TRAVESSA": "TRAVESSA", "TR": "TRAVESSA",
    "EST": "ESTRADA", "ESTR": "ESTRADA", "ESTRADA": "ESTRADA", "PC": "PRACA", "PCA": "PRACA",
    "PRACA": "PRACA", "LG": "LARGO", "LGO": "LARGO", "LARGO": "LARGO", "BC": "BECO",
    "BECO": "BECO", "AL": "ALAMEDA", "ALAMEDA": "ALAMEDA", "VL": "VILA", "VILA": "VILA",
    "ROD": "RODOVIA", "RODOVIA": "RODOVIA", "CGO": "CORREGO", "CORREGO": "CORREGO",
    "CAM": "CAMINHO", "CAMINHO": "CAMINHO", "PSG": "PASSAGEM", "PASSAGEM": "PASSAGEM",
    "LADEIRA": "LADEIRA", "LAD": "LADEIRA", "CAIS": "CAIS", "PATIO": "PATIO", "VIADUTO": "VIADUTO",
}
_TITULOS = {
    "DR": "DOUTOR", "DRA": "DOUTORA", "PROF": "PROFESSOR", "PROFA": "PROFESSORA",
    "STO": "SANTO", "STA": "SANTA", "S": "SAO", "PE": "PADRE", "GAL": "GENERAL",
    "GEN": "GENERAL", "CEL": "CORONEL", "CAP": "CAPITAO", "TEN": "TENENTE",
    "MAL": "MARECHAL", "ENG": "ENGENHEIRO", "DES": "DESEMBARGADOR", "PRES": "PRESIDENTE",
    "GOV": "GOVERNADOR", "DEP": "DEPUTADO", "SEN": "SENADOR", "VER": "VEREADOR",
    "MONS": "MONSENHOR", "CONS": "CONSELHEIRO", "COMEND": "COMENDADOR", "D": "DOM", "N": "NOSSA", "SRA": "SENHORA", "NS": "NOSSA SENHORA",
}


def normaliza_texto(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def nucleo_logradouro(nome: Optional[str]) -> str:
    """Nome da rua sem o tipo (RUA, AV, ...), com abreviações de título expandidas."""
    tokens = normaliza_texto(nome).split()
    while tokens and tokens[0] in _TIPOS:
        tokens = tokens[1:]
    # "1A TRAVESSA X" -> mantém o ordinal, mas descarta o tipo que vem depois.
    if len(tokens) > 1 and re.fullmatch(r"\d+A?", tokens[0]) and tokens[1] in _TIPOS:
        tokens = [tokens[0]] + tokens[2:]
    return " ".join(_TITULOS.get(t, t) for t in tokens)


class Geocodificador:
    """Rua + bairro -> ponto sobre a rua, dentro do bairro; senão, ponto no bairro.

    Níveis, do mais ao menos preciso:
    - `rua`: a rua da notificação foi encontrada entre as ruas daquele bairro;
      o ponto é sorteado ao longo dos trechos da rua que caem no bairro.
    - `rua_vizinha`: a rua não passa pelo bairro informado, mas por um vizinho
      (fichas às vezes trazem o bairro ao lado); ponto na rua, no vizinho.
    - `bairro`: o bairro foi reconhecido, a rua não; ponto uniforme no polígono.
    - `nenhum`: o bairro não foi reconhecido; o caso fica de fora.
    """

    def __init__(self, geo: Path = RAIZ / "geo", corte_similaridade: float = 0.88):
        import geopandas as gpd

        self.corte = corte_similaridade
        bairros = gpd.read_file(geo / "bairros.geojson").to_crs(CRS_RECIFE)
        bairros["chave"] = bairros["EBAIRRNOME"].map(normaliza_texto)
        self.bairros = bairros.set_index("chave")
        self.contorno = bairros.union_all()

        trechos = gpd.read_file(geo / "trechos_logradouros.geojson").to_crs(CRS_RECIFE)
        trechos = trechos.dropna(subset=["CLOGRACODI"])
        trechos["cod"] = trechos["CLOGRACODI"].astype(int)
        self._geom_rua = trechos.dissolve(by="cod")["geometry"]

        # A tabela oficial abrevia ("AV CONS AGUIAR") e também traz o nome por
        # extenso ("Avenida Conselheiro Aguiar"): as duas grafias viram chave.
        por_bairro = pd.read_csv(geo / "trechos_por_bairro.csv", sep=None, engine="python")
        por_bairro["chave"] = por_bairro["nomeBairro"].map(normaliza_texto)
        self.ruas: Dict[str, Dict[str, int]] = {}
        for b, g in por_bairro.groupby("chave"):
            ruas: Dict[str, int] = {}
            for coluna in ("nome_logradouro_concatenado", "nome_oficial_logradouro"):
                for nome, cod in zip(g[coluna], g["codlogradouro"].astype(int)):
                    nucleo = nucleo_logradouro(nome)
                    if nucleo:
                        ruas.setdefault(nucleo, cod)
            self.ruas[b] = ruas

        # Bairros vizinhos: fichas às vezes trazem o bairro ao lado de onde a
        # rua passa no mapa oficial.
        tocados = gpd.sjoin(bairros[["chave", "geometry"]],
                            bairros[["chave", "geometry"]].assign(geometry=bairros.buffer(50)),
                            predicate="intersects")
        self.vizinhos: Dict[str, list] = {
            b: sorted(set(g["chave_right"]) - {b}) for b, g in tocados.groupby("chave_left")
        }
        self._linha_cache: Dict[Tuple[int, str], object] = {}

    def bairro(self, nome: Optional[str]) -> Optional[str]:
        chave = normaliza_texto(nome)
        if not chave:
            return None
        if chave in self.bairros.index:
            return chave
        proximo = difflib.get_close_matches(chave, self.bairros.index, n=1, cutoff=self.corte)
        return proximo[0] if proximo else None

    def _rua_no_bairro(self, bairro: str, nucleo: str) -> Optional[int]:
        ruas = self.ruas.get(bairro)
        if not nucleo or not ruas:
            return None
        if nucleo in ruas:
            return ruas[nucleo]
        proximo = difflib.get_close_matches(nucleo, list(ruas), n=1, cutoff=self.corte)
        if proximo:
            return ruas[proximo[0]]
        # "AV SUL" -> "AV SUL GOVERNADOR CID SAMPAIO": prefixo, se for o único.
        if len(nucleo) >= 3:
            candidatos = {c for k, c in ruas.items() if k.startswith(nucleo + " ")}
            if len(candidatos) == 1:
                return candidatos.pop()
        return None

    def rua(self, bairro: str, logradouro: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
        """(código da rua, bairro onde ela foi achada): o informado ou um vizinho."""
        nucleo = nucleo_logradouro(logradouro)
        cod = self._rua_no_bairro(bairro, nucleo)
        if cod is not None:
            return cod, bairro
        for viz in self.vizinhos.get(bairro, []):
            cod = self._rua_no_bairro(viz, nucleo)
            if cod is not None:
                return cod, viz
        return None, None

    def _linha(self, cod: int, bairro: str):
        chave = (cod, bairro)
        if chave not in self._linha_cache:
            linha = None
            if cod in self._geom_rua.index:
                recorte = self._geom_rua.loc[cod].intersection(self.bairros.loc[bairro, "geometry"])
                linha = recorte if (not recorte.is_empty and recorte.length > 0) else None
            self._linha_cache[chave] = linha
        return self._linha_cache[chave]

    def _ponto_no_bairro(self, bairro: str, rng: np.random.Generator):
        from shapely.geometry import Point

        poli = self.bairros.loc[bairro, "geometry"]
        xmin, ymin, xmax, ymax = poli.bounds
        while True:
            p = Point(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
            if poli.contains(p):
                return p

    def geocodifica(self, df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        """Acrescenta `x`, `y` (metros, EPSG:31985), `bairro_oficial` e `nivel`."""
        rng = np.random.default_rng(seed)
        bairro_de = {n: self.bairro(n) for n in df["no_bairro_residencia"].dropna().unique()}
        rua_de: Dict[Tuple[str, str], Tuple[Optional[int], Optional[str]]] = {}
        xs, ys, niveis, oficiais = [], [], [], []
        for nome_b, logr in zip(df["no_bairro_residencia"], df["nome_logradouro_residencia"]):
            b = bairro_de.get(nome_b) if isinstance(nome_b, str) else None
            if b is None:
                xs.append(np.nan); ys.append(np.nan); niveis.append("nenhum"); oficiais.append(None)
                continue
            chave = (b, logr if isinstance(logr, str) else "")
            if chave not in rua_de:
                rua_de[chave] = self.rua(b, logr)
            cod, b_rua = rua_de[chave]
            linha = self._linha(cod, b_rua) if cod is not None else None
            if linha is not None:
                p = linha.interpolate(rng.uniform(0, linha.length))
                nivel = "rua" if b_rua == b else "rua_vizinha"
            else:
                p = self._ponto_no_bairro(b, rng)
                nivel = "bairro"
            xs.append(p.x); ys.append(p.y); niveis.append(nivel); oficiais.append(b)
        out = df.copy()
        out["x"], out["y"], out["nivel"], out["bairro_oficial"] = xs, ys, niveis, oficiais
        return out
