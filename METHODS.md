## Métodos de seleção de características implementados na ferramenta


| Método | Descrição | Utilidade |
|--------|------------|-----------|
| ABC (Artificial Bee Colony) | Algoritmo de otimização inspirado no comportamento de busca de alimento das abelhas. | Encontrar subconjuntos de características que melhoram a performance de um modelo. |
| ANOVA (Analysis of Variance) | Teste estatístico que analisa as diferenças entre médias de grupos. | Determinar quais características têm um impacto significativo nas variáveis de saída. |
| Qui-quadrado (Chi-Square Test) | Teste estatístico que avalia a independência entre duas variáveis categóricas. | Selecionar características que são estatisticamente significativas. |
| IG (Information Gain) | Medida baseada na teoria da informação para avaliar a importância de características. | Calcula a redução na entropia ao dividir os dados com base em uma característica. |
| LASSO (Least Absolute Shrinkage and Selection Operator) | Método de regressão que usa regularização L1 para induzir esparsidade no modelo. | Selecionar características importantes ao reduzir os coeficientes das menos relevantes a zero. |
| LR (Logistic Regression) | Modelo estatístico usado para prever a probabilidade de um evento binário. | Avaliar a importância das características pelos coeficientes do modelo. |
| MAD (Median Absolute Deviation) | Medida de variabilidade robusta baseada na mediana. | Características com baixa variabilidade são consideradas menos informativas e podem ser descartadas. |
| PCA (Principal Component Analysis) | Técnica de redução de dimensionalidade que transforma características originais em componentes principais. | Identificar combinações lineares de características que explicam a maior parte da variabilidade nos dados. |
| PCC (Pearson Correlation Coefficient) | Medida da correlação linear entre duas variáveis. | Características com alta correlação com a variável de saída são selecionadas. |
| ReliefF | Algoritmo que estima a relevância das características com base na habilidade de distinguir entre instâncias próximas. | Considera a relação entre características e o rótulo da classe, bem como a redundância entre características. |
| RFE (Recursive Feature Elimination) | Técnica de seleção de características que remove recursivamente as menos importantes. | Utiliza um modelo preditivo para avaliar a importância das características. |
| Variancia | Método que seleciona características com base na variância. | Características com baixa variância são removidas, pois têm pouca informação para a tarefa de classificação. |
| JOWNDroid | Método específico para seleção de características em datasets de malware Android. | Foca em características que diferenciam entre malware e aplicativos benignos. |
| MT (Multi-Tiered) | Estrutura de dados que organiza características de acordo com sua métrica de similaridade. | Identificar características que são consistentes em diferentes instâncias de dados. |
| RFG (Random Forest-based Feature Selection) | Método baseado em Random Forest para avaliar a importância das características. | Utiliza o conceito de redução da impureza em árvores de decisão para selecionar características relevantes. |
| SemiDroid | Método específico para análise de malware em dispositivos Android. | Combina técnicas de análise estática e dinâmica para selecionar características discriminantes. |
| SigAPI | Seleção de características baseada em chamadas de API significativas utilizadas por aplicativos Android. | Identificar chamadas de API que são fortemente associadas a comportamentos maliciosos. |
| SigPID | Similar ao SIGAPI, mas se concentra em identificar permissões significativos. | Detectar comportamentos anômalos em processos de aplicativos Android. |
