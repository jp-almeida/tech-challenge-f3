# Roteiro do vídeo — modelo STAR

**Duração estimada: 4:31** (limite de 5:00) · 609 palavras faladas a ~135 palavras/minuto.
A margem de 29 segundos cobre as pausas entre os blocos.

Como usar: o texto em **citação** é para ler em voz alta, do jeito que está. As linhas
`🎬 Tela:` são o que mostrar enquanto fala — o vídeo inteiro percorre o README do repositório.

Antes de gravar: abra o README já rolado no topo, deixe o Grafana e o Swagger em abas separadas,
aumente a fonte do navegador e ensaie uma vez cronometrando. Se passar de 4:50, corte o parágrafo
marcado como **[corte opcional]** — ele devolve 13 segundos. Fale em ritmo normal: ler rápido para
encaixar mais conteúdo custa mais nota do que cortar um parágrafo.

| Bloco | Tempo | Duração |
|---|---|---|
| **S** — Situation | 0:00 – 0:40 | 40s |
| **T** — Task | 0:40 – 1:10 | 30s |
| **A** — Action | 1:10 – 3:05 | 1min55 |
| **R** — Result | 3:05 – 4:25 | 1min20 |
| Lições e fecho | 4:25 – 4:40 | 15s |

---

## S — Situation (0:00 – 0:40)

🎬 **Tela:** topo do README (título e badge do CI), depois a seção **1. Contexto e problema**.

> Num hospital, os laudos são lidos por ordem de chegada. O problema é que a fila não distingue um
> achado sem gravidade de um infarto em curso.

> Quando um caso tempo-dependente — um AVC, uma síndrome coronariana aguda — espera horas por
> leitura, a janela de tratamento vai se fechando. Não é um problema de volume: é de ordenação.
> A informação para priorizar já está no texto do laudo.

---

## T — Task (0:40 – 1:10)

🎬 **Tela:** ainda na seção 1, destacando a frase do requisito; depois o diagrama da seção **2. Arquitetura local**.

> A tarefa foi construir a esteira completa de MLOps para priorizar essa fila automaticamente:
> classificar cada laudo em normal, atenção ou urgente.

> E entregar como um sistema que roda de verdade: API em container, CI/CD, retreino orquestrado,
> monitoramento e otimização de latência. Um requisito guiou todas as decisões — um laudo urgente
> precisa ser sinalizado em segundos, não na próxima janela de processamento.

---

## A — Action (1:10 – 3:05)

🎬 **Tela:** seção **4. Dados e mapeamento de rótulos**, parando na tabela do mapeamento.

> Começamos pelos dados: catorze mil resumos médicos rotulados por especialidade. Como não há
> rótulo de urgência, mapeamos cinco categorias em três níveis — neurológico e cardiovascular
> viram urgente; neoplasias e digestivo viram atenção; o restante, normal.

> Isso é um proxy, não urgência clínica aferida — e está declarado como limitação no README, não
> escondido no rodapé.

🎬 **Tela:** seção **5. Modelo e métricas**, depois o diagrama da arquitetura.

> O modelo é TF-IDF com regressão logística: leve, interpretável e rápido em CPU. Ele é servido por
> uma API FastAPI rodando ONNX Runtime.

> Uma decisão de arquitetura vale destacar: toda a lógica de treino vive em funções Python puras,
> testadas sem Airflow. A DAG é só um invólucro fino. Na prática, o retreino roda igual pelo
> Makefile ou pelo orquestrador — e uma falha do Airflow nunca o bloqueia.

🎬 **Tela:** seção **9. Orquestração**, mostrando o print do grafo da DAG.

> No Airflow, a DAG tem seis tasks: ingestão, treino, avaliação, quality gate, exportação para ONNX
> e promoção. O quality gate é o detalhe que importa: se o F1 cair abaixo do limiar, o run falha e o
> modelo não é promovido. Testamos isso forçando o limiar para 0,99 — a promoção foi bloqueada e os
> artefatos ficaram intactos.

🎬 **Tela:** seção **10. Monitoramento**, depois **8. CI/CD** com o print do Actions.

> A API se instrumenta com prometheus_client. O Prometheus raspa as métricas a cada cinco segundos e
> o Grafana sobe com o dashboard já provisionado, sem nenhum clique.

> **[corte opcional — 30 palavras, ~13s]** O CI roda lint, testes e build com smoke test. O smoke
> test garante que a imagem realmente serve predição, não só que o build passou.

---

## R — Result (3:05 – 4:25)

🎬 **Tela:** tabela de métricas da seção 5.

> Nos resultados. O modelo atinge F1 macro de 0,61, contra uma linha de base aleatória de 0,33. É
> sinal real, mas não sustenta decisão clínica autônoma — e o README afirma isso com todas as letras.

🎬 **Tela:** seção **6. Otimização e resultados de latência**, com o gráfico e a tabela.

> A técnica de otimização foi a conversão para ONNX Runtime. In-process, a inferência ficou duas
> vírgula três vezes mais rápida: de vinte e dois centésimos para nove centésimos de milissegundo.

> Ponta a ponta, via HTTP, o ganho cai para um vírgula três. Essa diferença é o achado mais
> interessante do projeto: depois do ONNX, mais de noventa por cento do tempo da requisição é
> overhead de rede e de JSON, não inferência. O modelo deixou de ser o gargalo — e preferimos
> mostrar isso a exibir só o número mais bonito.

> A conversão foi validada: cem por cento de concordância de rótulos nos dois mil oitocentos e
> oitenta e oito textos de teste.

🎬 **Tela:** seção **7. Como executar**, depois o dashboard do Grafana com os painéis se movendo.

> Para fechar: a stack inteira sobe em três comandos. Testamos num clone limpo — treze segundos até
> tudo de pé. São 48 testes automatizados, CI verde, e o dashboard com seis painéis mostrando
> requisições por segundo, latência, taxa de erro e a distribuição das predições por classe.

---

## Lições e fecho (4:25 – 4:40)

🎬 **Tela:** seção **13. Limitações, riscos e próximos passos**; terminar no topo do README com o link do repositório.

> Duas lições. Dois bugs só apareceram dentro do container — o ONNX exigia um locale que a imagem
> slim não tinha. Rodar na própria máquina não é evidência de que funciona.

> E otimizar o modelo só compensa até ele deixar de ser o gargalo. Obrigado.

---

## Checklist antes de publicar

- [ ] Duração final **≤ 5:00**
- [ ] Números falados batem com o README (F1 0,61 · 2,3× · 1,34× · 100% · 13s · 48 testes)
- [ ] Áudio sem ruído e fonte do terminal/navegador legível em tela cheia
- [ ] Vídeo publicado como **não listado** e link testado em aba anônima
- [ ] Link adicionado ao topo do README e à plataforma de entrega
