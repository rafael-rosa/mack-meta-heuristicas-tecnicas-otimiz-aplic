import pandas as pd
import ast
import re
from collections import defaultdict

# --- Funções de Carregamento e Pré-processamento ---

def carregar_estrutura_tempo(filepath="meta_heuristicas\projeto_final\datasets\estrutura_tempo.txt"):
    """
    Carrega e processa o arquivo estrutura_tempo.txt.
    Retorna um mapa de string de tempo para índice e de índice para string de tempo,
    e o número total de blocos.
    """
    idx_para_str_tempo = {}
    str_tempo_para_idx = {}
    total_blocos = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        mapa_content = content.split('MAPA_BLOCO_INDEX_PARA_TEMPO:\n')[-1]
        for line in mapa_content.splitlines():
            if line.strip():
                match = re.match(r"(\d+):\s*(.*)", line.strip())
                if match:
                    idx = int(match.group(1))
                    tempo_str = match.group(2).strip()
                    idx_para_str_tempo[idx] = tempo_str
                    str_tempo_para_idx[tempo_str] = idx
                    total_blocos = max(total_blocos, idx + 1)
    except FileNotFoundError:
        print(f"Erro: Arquivo {filepath} não encontrado.")
        return None, None, 0
    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")
        return None, None, 0
    if not idx_para_str_tempo:
        print(f"Erro: Nenhum mapeamento de tempo encontrado em {filepath}.")
        return None, None, 0
    return idx_para_str_tempo, str_tempo_para_idx, total_blocos

def carregar_professores(filepath="meta_heuristicas\projeto_final\datasets\professores.csv", str_tempo_para_idx=None):
    """
    Carrega professores.csv e processa indisponibilidades.
    Retorna um DataFrame e um dicionário de indisponibilidade por professor (índices).
    """
    if str_tempo_para_idx is None:
        print("Erro: Mapa str_tempo_para_idx é necessário para carregar professores.")
        return None, None
    try:
        df_professores = pd.read_csv(filepath)
        mapa_indisponibilidade_prof = {}
        for _, row in df_professores.iterrows():
            prof_id = row['id']
            try:
                blocos_str_lista = ast.literal_eval(row['indisponibilidade_blocos'])
                indices_indisponiveis = set()
                for bloco_str in blocos_str_lista:
                    if bloco_str in str_tempo_para_idx:
                        indices_indisponiveis.add(str_tempo_para_idx[bloco_str])
                    else:
                        print(f"Aviso: Bloco de tempo '{bloco_str}' na indisponibilidade do professor {prof_id} não reconhecido.")
                mapa_indisponibilidade_prof[prof_id] = indices_indisponiveis
            except (ValueError, SyntaxError) as e:
                print(f"Aviso: Erro ao processar indisponibilidade para professor {prof_id}: {e}. Bloco: {row['indisponibilidade_blocos']}")
                mapa_indisponibilidade_prof[prof_id] = set() # Considerar sem indisponibilidades se houver erro

        return df_professores, mapa_indisponibilidade_prof
    except FileNotFoundError:
        print(f"Erro: Arquivo {filepath} não encontrado.")
        return None, None
    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")
        return None, None

def carregar_salas(filepath="meta_heuristicas\projeto_final\datasets\salas.csv"):
    """Carrega salas.csv."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Erro: Arquivo {filepath} não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")
        return None

def carregar_disciplinas_turmas(filepath="meta_heuristicas\projeto_final\datasets\disciplinas_turmas.csv"):
    """Carrega disciplinas_turmas.csv e cria mapa de duração."""
    try:
        df_disciplinas = pd.read_csv(filepath)
        mapa_duracao_turma = pd.Series(df_disciplinas.duracao_blocos.values, index=df_disciplinas.id_turma).to_dict()
        return df_disciplinas, mapa_duracao_turma
    except FileNotFoundError:
        print(f"Erro: Arquivo {filepath} não encontrado.")
        return None, None
    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")
        return None, None

def analisar_grade_horaria(filepath="meta_heuristicas\projeto_final\saida\grade_horaria_final.txt"):
    """Analisa o arquivo de grade horária e extrai as alocações."""
    alocacoes = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex para capturar os dados de cada turma
        # Ajustado para capturar IDs corretamente e o slot numérico
        padrao_turma = re.compile(
            r"Turma:\s*([A-Z0-9_]+)\s*\(.*?\)\s*\n"  # ID da Turma
            r"\s*Professor:\s*.*?\s*\((P\d+)\)\s*\n" # ID do Professor
            r"\s*Sala:\s*.*?\s*\((S\d+)\)\s*\n"      # ID da Sala
            r"\s*Horario:\s*Slot\s*(\d+)\s*\(.*?\)\s*for\s*(\d+)\s*blocos" # Slot e Duração
        )
        
        matches = padrao_turma.finditer(content)
        for match in matches:
            id_turma = match.group(1)
            id_prof = match.group(2)
            id_sala = match.group(3)
            bloco_inicio = int(match.group(4))
            # A duração no arquivo de grade é informativa, usaremos a do disciplinas_turmas.csv
            # duracao_grade = int(match.group(5)) 
            
            alocacoes.append({
                "id_turma": id_turma,
                "id_prof": id_prof,
                "id_sala": id_sala,
                "bloco_inicio": bloco_inicio
            })
            
    except FileNotFoundError:
        print(f"Erro: Arquivo {filepath} não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")
        return None
    if not alocacoes:
        print(f"Aviso: Nenhuma alocação encontrada em {filepath}.")
    return alocacoes

# --- Função Principal de Validação ---

def validar_grade_horaria():
    print("Iniciando validação da grade horária...\n")

    # 1. Carregar todos os dados de entrada
    idx_para_str_tempo, str_tempo_para_idx, total_blocos = carregar_estrutura_tempo()
    if idx_para_str_tempo is None: return

    df_professores, mapa_indisponibilidade_prof = carregar_professores(str_tempo_para_idx=str_tempo_para_idx)
    if df_professores is None: return

    df_salas = carregar_salas()
    if df_salas is None: return

    df_disciplinas, mapa_duracao_turma = carregar_disciplinas_turmas()
    if df_disciplinas is None: return

    # 2. Analisar o arquivo da grade horária
    alocacoes_grade = analisar_grade_horaria()
    if not alocacoes_grade: # Se for None ou lista vazia
        print("Validação não pode prosseguir sem alocações da grade.")
        return

    print(f"Total de {len(alocacoes_grade)} alocações encontradas na grade para validar.\n")

    # 3. Realizar as Validações
    conflitos_encontrados = []
    
    # Estruturas para rastrear ocupação de professores e salas
    # {id_recurso: {bloco_idx: id_turma_ocupando}}
    horario_professores_ocupado = defaultdict(dict) 
    horario_salas_ocupado = defaultdict(dict)

    for alocacao in alocacoes_grade:
        id_turma = alocacao["id_turma"]
        id_prof = alocacao["id_prof"]
        id_sala = alocacao["id_sala"]
        bloco_inicio = alocacao["bloco_inicio"]

        if id_turma not in mapa_duracao_turma:
            msg = f"CONFLITO DE DADOS: Turma '{id_turma}' da grade não encontrada no arquivo de disciplinas."
            print(msg)
            conflitos_encontrados.append(msg)
            continue # Pula para a próxima alocação se a turma não existe

        duracao = mapa_duracao_turma[id_turma]
        blocos_ocupados_pela_aula = set(range(bloco_inicio, bloco_inicio + duracao))

        # Verificar se os blocos são válidos
        for bloco in blocos_ocupados_pela_aula:
            if not (0 <= bloco < total_blocos):
                msg = (f"CONFLITO DE HORÁRIO INVÁLIDO para Turma '{id_turma}': Bloco {bloco} "
                       f"(início: {bloco_inicio}, duração: {duracao}) está fora do intervalo de tempo válido [0-{total_blocos-1}].")
                print(msg)
                conflitos_encontrados.append(msg)
                # Não precisa continuar checando esta alocação se o bloco é inválido
                break 
        if any(not (0 <= bloco < total_blocos) for bloco in blocos_ocupados_pela_aula):
            continue


        # A. Validar Indisponibilidade Explícita do Professor
        indisponibilidades_prof_especifico = mapa_indisponibilidade_prof.get(id_prof, set())
        conflito_indisponibilidade_explicita = blocos_ocupados_pela_aula.intersection(indisponibilidades_prof_especifico)
        
        if conflito_indisponibilidade_explicita:
            blocos_conflitantes_str = [idx_para_str_tempo.get(b, str(b)) for b in sorted(list(conflito_indisponibilidade_explicita))]
            msg = (f"CONFLITO DE PROFESSOR (Indisponibilidade Explícita): Turma '{id_turma}' alocada ao Prof. '{id_prof}' "
                   f"durante seu(s) horário(s) de indisponibilidade: {', '.join(blocos_conflitantes_str)}.")
            print(msg)
            conflitos_encontrados.append(msg)

        # B. Validar Conflito de Professor (Professor em duas aulas ao mesmo tempo)
        for bloco_idx in blocos_ocupados_pela_aula:
            if bloco_idx in horario_professores_ocupado.get(id_prof, {}):
                turma_conflitante = horario_professores_ocupado[id_prof][bloco_idx]
                if turma_conflitante != id_turma: # Evita auto-conflito se processar a mesma turma de novo por algum motivo
                    bloco_str = idx_para_str_tempo.get(bloco_idx, str(bloco_idx))
                    msg = (f"CONFLITO DE PROFESSOR (Dupla Alocação): Prof. '{id_prof}' está alocado para Turma '{id_turma}' "
                           f"E Turma '{turma_conflitante}' simultaneamente no bloco: {bloco_str} (Índice {bloco_idx}).")
                    print(msg)
                    conflitos_encontrados.append(msg)
            else:
                # Marcar ocupação do professor se não houver conflito ainda para este bloco
                if id_prof not in horario_professores_ocupado:
                    horario_professores_ocupado[id_prof] = {}
                horario_professores_ocupado[id_prof][bloco_idx] = id_turma
        
        # C. Validar Conflito de Sala (Sala com duas aulas ao mesmo tempo)
        for bloco_idx in blocos_ocupados_pela_aula:
            if bloco_idx in horario_salas_ocupado.get(id_sala, {}):
                turma_conflitante = horario_salas_ocupado[id_sala][bloco_idx]
                if turma_conflitante != id_turma:
                    bloco_str = idx_para_str_tempo.get(bloco_idx, str(bloco_idx))
                    msg = (f"CONFLITO DE SALA (Dupla Alocação): Sala '{id_sala}' está alocada para Turma '{id_turma}' "
                           f"E Turma '{turma_conflitante}' simultaneamente no bloco: {bloco_str} (Índice {bloco_idx}).")
                    print(msg)
                    conflitos_encontrados.append(msg)
            else:
                # Marcar ocupação da sala
                if id_sala not in horario_salas_ocupado:
                    horario_salas_ocupado[id_sala] = {}
                horario_salas_ocupado[id_sala][bloco_idx] = id_turma

    # 4. Relatar Resultados
    print("\n--- Relatório Final da Validação ---")
    if not conflitos_encontrados:
        print("NENHUM CONFLITO ENCONTRADO! A grade horária parece válida em relação às restrições verificadas.")
    else:
        print(f"Total de {len(conflitos_encontrados)} conflitos ou avisos encontrados:")
        # Os conflitos já foram impressos quando detectados.
        # Se quiser um resumo aqui, pode iterar sobre a lista conflitos_encontrados.
        # for i, conflito in enumerate(conflitos_encontrados):
        # print(f"  {i+1}. {conflito}")

if __name__ == '__main__':
    validar_grade_horaria()