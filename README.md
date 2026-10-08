# Laboratorio ORB — OpenCV + Python

Implementacao didatica que **usa a implementacao pronta** `cv.ORB_create()` do OpenCV. Nao recria manualmente FAST nem o conjunto de testes rBRIEF.

## Estrutura

```text
orb_opencv_lab/
  data/input/      # Imagens de entrada
  data/output/     # Resultados gerados
  src/orb_experimento.py
  scripts/gerar_exemplo.py
  scripts/avaliacao_lote.py
  tests/test_orb.py
  requirements.txt
```

## Ambiente Windows (PowerShell), a partir da pasta do projeto

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se a politica de execucao bloquear a ativacao, use o interpretador diretamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/gerar_exemplo.py
```

## Execucao (na raiz do projeto)

```powershell
python scripts/gerar_exemplo.py
python src/orb_experimento.py --image data/input/exemplo.png
python src/orb_experimento.py --image data/input/exemplo.png --rotate 45
python src/orb_experimento.py --image data/input/exemplo.png --rotate 45 --noise 10
python src/orb_experimento.py --image data/input/foto1.jpg --compare data/input/foto2.jpg
python scripts/avaliacao_lote.py --image data/input/exemplo.png
python -m unittest discover -s tests -v
```

### Parametros uteis

- `--nfeatures 1000`: limite pretendido de keypoints; o numero efetivo pode ser menor.
- `--ratio 0.75`: limite do teste da razao entre 1o e 2o descritor mais proximo.
- `--rotate 45`: gera segunda imagem girada; as bordas podem ser recortadas.
- `--noise 10`: aplica ruido gaussiano somente na imagem girada.
- `--output data/output/ensaio45`: caminho de saida personalizado, para nao sobrescrever ensaios.

## Saidas

- `imagem1_keypoints.png` e `imagem2_keypoints.png`: pontos, tamanhos e orientacoes.
- `imagem1_keypoints.csv` e `imagem2_keypoints.csv`: propriedades dos pontos.
- `imagem1_descriptors.npy` e `imagem2_descriptors.npy`: arrays binarios compactados, usualmente `(N, 32)`.
- `correspondencias.png`: ate 60 matches apos teste da razao.
- `imagem2_transformada.png`: imagem gerada com rotacao/ruido (quando utilizada).
- `resultados.json`: contagens, tempos e indicadores.
- `avaliacao_lote.csv`: varredura de angulos e ruido com medidas repetidas, criada pelo script de lote.

A distancia entre descritores ORB com `WTA_K=2` e Hamming. Os 32 bytes por descritor codificam 256 bits. A relacao de Lowe e uma heuristica adicional de filtragem, nao uma etapa do detector ORB.

O campo `precisao_geometrica_pct` e calculado **somente para rotacao sintetica**, comparando coordenadas pareadas com a transformacao verdadeira e limiar de erro de 3 pixels. Isso mede a precisao dos matches filtrados, **nao recall** ou repetibilidade de todos os keypoints. O RANSAC e apenas diagnostico e a homografia e mais apropriada para objetos ou cenas aproximadamente planas; sua taxa de inliers **nao equivale a verdade-terreno**.

A avaliacao em lote preserva a mesma imagem-base e a transformacao geometricamente conhecida, mas o recorte e os pixels pretos nas bordas causados por `warpAffine` tambem afetam o resultado. Replicacoes com sigma=0 sao identicas quanto a imagem; outras repeticoes usam sementes diferentes.

Uma unica medicao de tempo nao e benchmark: para estudos quantitativos, rode multiplos ensaios com aquecimento, hardware fixo e estatisticas agregadas. O desempenho do artigo de 2011 nao e comparavel diretamente a uma maquina moderna.

## Referencias

- Rublee, E. et al. ORB: an efficient alternative to SIFT or SURF. ICCV, 2011.
- OpenCV ORB: https://docs.opencv.org/5.0/py_tutorials/py_features/py_orb/py_orb.html
- OpenCV Feature Matching: https://docs.opencv.org/5.0/py_tutorials/py_features/py_matcher/py_matcher.html
