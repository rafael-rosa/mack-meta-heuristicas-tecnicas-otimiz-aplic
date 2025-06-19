# Resolvedor de Grade Horária Universitária com Algoritmo Genético

## 1. Descrição do Projeto

Este projeto apresenta uma solução computacional para o **Problema de Escalonamento de Grades Horárias Universitárias (UTP - University Timetabling Problem)**, um desafio clássico de otimização combinatória. O objetivo é alocar automaticamente um conjunto de turmas a professores, salas e horários, respeitando um conjunto complexo de restrições rígidas (obrigatórias) e flexíveis (preferenciais).

A solução foi desenvolvida como parte do projeto final da disciplina de Meta-heurísticas e Técnicas de Otimização Aplicadas.

## 2. Metodologia

A abordagem para a resolução do problema é **híbrida**, combinando duas fases principais para garantir tanto a factibilidade quanto a qualidade da grade horária gerada:

1.  **Heurística Construtiva:** Primeiramente, uma heurística gulosa é executada para gerar uma solução inicial. Ela ordena as turmas por um critério de dificuldade e as aloca sequencialmente na primeira combinação válida de recursos encontrada. O objetivo desta fase é obter rapidamente uma grade horária completa e factível, que servirá de ponto de partida para a otimização.

2.  **Algoritmo Genético (AG):** A solução inicial é então refinada por um Algoritmo Genético. O AG evolui uma população de grades horárias ao longo de várias gerações, utilizando operadores de **seleção**, **cruzamento (crossover)** e **mutação**. A qualidade de cada solução é medida por uma **função de aptidão (fitness)** que penaliza severamente a violação de restrições rígidas (ex: conflito de horários, professor não compatível, sala sem recursos/capacidade). Este processo de otimização busca encontrar uma solução globalmente superior e 100% válida.

## 3. Estrutura do Repositório
```bash
.
├── datasets/
│   ├── disciplinas_turmas.csv
│   ├── professores.csv
│   ├── salas.csv
│   └── estrutura_tempo.txt
│
├── saida/
│   ├── grade_horaria_final.txt
│
├── solver_v2.1.py
├── validador_grade_horaria.py
├── grade_horaria_final.txt
└── README.md
```

* **`datasets/`**: Pasta contendo todos os arquivos de dados de entrada necessários para a execução do problema.
* **`solver_v2.1.py`**: O script principal que implementa a heurística construtiva e o Algoritmo Genético para resolver o problema e gerar a grade horária.
* **`validador_grade_horaria.py`**: Um script utilitário para verificar se a grade horária gerada (`grade_horaria_final.txt`) respeita todas as restrições rígidas.
* **`grade_horaria_final.txt`**: Arquivo de saída gerado pelo `solver_v2.1.py`, contendo a melhor grade horária encontrada.
* **`README.md`**: Este arquivo.

## 4. Como Executar o Projeto

### Pré-requisitos

* Python 3.x
* Biblioteca Pandas (`pip install pandas`)

### Passos para Execução

**Passo 1: Configurar os Dados de Entrada**

Certifique-se de que a pasta `datasets/` existe na raiz do projeto e contém os quatro arquivos de entrada:
* `disciplinas_turmas.csv`
* `professores.csv`
* `salas.csv`
* `estrutura_tempo.txt`

**Passo 2: Executar o Resolvedor**

Para gerar a grade horária, execute o script principal a partir do seu terminal:

```bash
python solver_v2.py
```

O script irá carregar os dados, executar a heurística e o Algoritmo Genético, e ao final, salvará a melhor solução encontrada no arquivo grade_horaria_final.txt. 

**Passo 3: Validar a Grade Gerada**

Após a execução do solver_v2.py, você pode verificar a validade da solução gerada executando o script de validação:

```bash
python validador_grade_horaria.py
```

Este script lerá os arquivos de entrada e o arquivo grade_horaria_final.txt, e imprimirá no console um relatório indicando se foram encontrados conflitos de restrições rígidas.

## 5. Autores

* **José Augusto Lima**
* **Rafael da Silva Rosa**
