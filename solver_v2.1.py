import pandas as pd
import random
import ast  # Para avaliar com segurança strings que representam listas
from collections import defaultdict
import copy # Para cópia profunda de indivíduos (soluções) no AG
import time

# 1. Grava o tempo de início
start_time = time.time()

# --- Configuração e Constantes ---
# Parâmetros do AG (podem ser ajustados)
POPULATION_SIZE = 50
MAX_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 2 # Número de melhores indivíduos a serem mantidos na próxima geração

# Penalidades da Função de Aptidão (podem ser ajustadas)
UNSCHEDULED_CLASS_PENALTY = 1000  # Penalidade alta para cada turma não alocada

# Esta penalidade deve ser muito alta, pois é uma restrição rígida.
# Um valor maior que UNSCHEDULED_CLASS_PENALTY incentiva o algoritmo a preferir
# deixar uma aula sem alocar a alocá-la em um horário inválido para o professor.
PROFESSOR_UNAVAILABLE_PENALTY = 2500


# --- 1. Carregamento e Pré-processamento dos Dados ---

def load_and_preprocess_data():
    """Carrega e pré-processa todos os arquivos de dados."""
    print("Carregando e pré-processando os dados...")

    # Carrega os arquivos CSV
    try:
        # Ajuste os caminhos dos arquivos conforme a sua estrutura de pastas
        turmas_df = pd.read_csv("meta_heuristicas\projeto_final\datasets\disciplinas_turmas.csv")
        professores_df = pd.read_csv("meta_heuristicas\projeto_final\datasets\professores.csv")
        salas_df = pd.read_csv("meta_heuristicas\projeto_final\datasets\salas.csv")
    except FileNotFoundError as e:
        print(f"Erro: {e}. Certifique-se de que todos os arquivos CSV estão no mesmo diretório.")
        return None

    # Converte strings de listas em listas/conjuntos reais
    def parse_string_list(s):
        try:
            parsed_list = ast.literal_eval(s)
            return set(parsed_list) if isinstance(parsed_list, list) else parsed_list
        except (ValueError, SyntaxError):
            return set() # Retorna um conjunto vazio para strings malformadas

    turmas_df['prof_compativel_ids'] = turmas_df['prof_compativel_ids'].apply(parse_string_list)
    turmas_df['recursos_necessarios'] = turmas_df['recursos_necessarios'].apply(parse_string_list)

    professores_df['indisponibilidade_blocos'] = professores_df['indisponibilidade_blocos'].apply(parse_string_list)
    salas_df['recursos'] = salas_df['recursos'].apply(parse_string_list)

    # Carrega e processa o arquivo estrutura_tempo.txt
    time_slots_map = {} # índice -> "Dia HH:MM-HH:MM"
    try:
        with open("meta_heuristicas\projeto_final\datasets\estrutura_tempo.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        
        mapa_content = content.split('MAPA_BLOCO_INDEX_PARA_TEMPO:\n')[-1]
        for line in mapa_content.splitlines():
            if line.strip():
                idx_str, time_str = line.split(':', 1)
                time_slots_map[int(idx_str.strip())] = time_str.strip()
    except FileNotFoundError:
        print("Erro: arquivo estrutura_tempo.txt não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao processar estrutura_tempo.txt: {e}")
        return None

    if not time_slots_map:
        print("Erro: O mapa de blocos de tempo está vazio. Verifique o formato de estrutura_tempo.txt.")
        return None
        
    total_time_slots = len(time_slots_map)

    # Cria o mapa reverso para os blocos de tempo (Dia HH:MM-HH:MM -> índice)
    time_slots_reverse_map = {v: k for k, v in time_slots_map.items()}

    def convert_prof_indisponibilidade(ind_set_str, reverse_map):
        ind_indices = set()
        for time_str in ind_set_str:
            if time_str in reverse_map:
                ind_indices.add(reverse_map[time_str])
        return ind_indices

    professores_df['indisponibilidade_indices'] = professores_df['indisponibilidade_blocos'].apply(
        lambda x: convert_prof_indisponibilidade(x, time_slots_reverse_map)
    )
    
    print("Dados carregados e pré-processados com sucesso.")
    return {
        "turmas": turmas_df,
        "professores": professores_df,
        "salas": salas_df,
        "time_slots_map": time_slots_map,
        "time_slots_reverse_map": time_slots_reverse_map,
        "total_time_slots": total_time_slots
    }

# --- 2. Funções de Verificação de Restrições e Gestão de Horários ---

def is_slot_sequence_valid(start_slot, duration, total_slots):
    """Verifica se a sequência de blocos de tempo está dentro dos limites."""
    return start_slot + duration <= total_slots

def check_professor_availability(prof_id, start_slot, duration, professor_schedule, prof_indisponibility_indices):
    """Verifica se o professor está disponível e não em seus horários de indisponibilidade."""
    for i in range(duration):
        current_slot = start_slot + i
        if current_slot in prof_indisponibility_indices: return False # O professor tem uma indisponibilidade fixa
        if prof_id in professor_schedule and current_slot in professor_schedule[prof_id]: return False # O professor já está alocado em outra aula
    return True

def check_room_availability(room_id, start_slot, duration, room_schedule):
    """Verifica se a sala está disponível."""
    for i in range(duration):
        current_slot = start_slot + i
        if room_id in room_schedule and current_slot in room_schedule[room_id]: return False # A sala já está alocada
    return True

def update_schedules(timetable_entry, professor_schedule, room_schedule):
    """Marca professor e sala como ocupados nos blocos da aula agendada."""
    turma = timetable_entry['turma_data']
    prof_id = timetable_entry['professor_id']
    room_id = timetable_entry['sala_id']
    start_slot = timetable_entry['bloco_inicio']
    duration = turma['duracao_blocos']
    if prof_id not in professor_schedule: professor_schedule[prof_id] = set()
    if room_id not in room_schedule: room_schedule[room_id] = set()
    for i in range(duration):
        professor_schedule[prof_id].add(start_slot + i)
        room_schedule[room_id].add(start_slot + i)

def clear_schedules_for_class(turma_data, assignment, professor_schedule, room_schedule):
    """Libera os horários do professor e da sala para uma dada alocação de aula."""
    if not assignment or assignment['professor_id'] is None: return
    prof_id = assignment['professor_id']
    room_id = assignment['sala_id']
    start_slot = assignment['bloco_inicio']
    duration = turma_data['duracao_blocos']
    if prof_id in professor_schedule:
        for i in range(duration): professor_schedule[prof_id].discard(start_slot + i)
    if room_id in room_schedule:
        for i in range(duration): room_schedule[room_id].discard(start_slot + i)


# --- 3. Heurística Construtiva ---

def constructive_heuristic(data):
    """Gera uma grade horária inicial usando uma heurística construtiva."""
    print("Executando a heurística construtiva...")
    turmas_df = data['turmas'].copy()
    professores_df = data['professores']
    salas_df = data['salas']
    total_time_slots = data['total_time_slots']
    
    # Ordena as turmas (exemplo: por menos profs. compatíveis, depois duração, depois alunos)
    turmas_df['num_compatible_profs'] = turmas_df['prof_compativel_ids'].apply(len)
    turmas_df = turmas_df.sort_values(by=['num_compatible_profs', 'duracao_blocos', 'alunos_estimados'], ascending=[True, False, False])
    timetable = []
    unscheduled_turmas_ids = []
    
    # horario_professores: {id_prof: {indices_dos_blocos}}
    # horario_salas: {id_sala: {indices_dos_blocos}}
    professor_schedule = defaultdict(set)
    room_schedule = defaultdict(set)
    
    # Preenche o horário dos professores com suas indisponibilidades fixas
    for _, prof_row in professores_df.iterrows():
        professor_schedule[prof_row['id']] = prof_row['indisponibilidade_indices'].copy()
    for _, turma_row in turmas_df.iterrows():
        turma_id = turma_row['id_turma']
        scheduled_this_turma = False
        potential_professors = [p for p in professores_df.itertuples() if turma_row['prof_compativel_ids'] and p.id in turma_row['prof_compativel_ids']]
        potential_salas = list(salas_df.itertuples())
        
        # Ordenar salas (ex: menor capacidade que sirva, para poupar salas maiores)
        potential_salas.sort(key=lambda s: s.capacidade)

        for prof in potential_professors:
            if scheduled_this_turma: break
            for sala in potential_salas:
                if scheduled_this_turma: break
                
                # Verifica a compatibilidade básica da sala
                if not (turma_row['recursos_necessarios'].issubset(sala.recursos) and turma_row['alunos_estimados'] <= sala.capacidade): continue
                
                for start_slot in range(total_time_slots):
                    if not is_slot_sequence_valid(start_slot, turma_row['duracao_blocos'], total_time_slots): continue
                    
                    # Verifica todas as restrições rígidas
                    prof_available = check_professor_availability(prof.id, start_slot, turma_row['duracao_blocos'], professor_schedule, prof.indisponibilidade_indices)
                    if not prof_available: continue
                    
                    room_available = check_room_availability(sala.id, start_slot, turma_row['duracao_blocos'], room_schedule)
                    if not room_available: continue
                    
                    # Se todas as verificações passarem, aloca a aula
                    assignment = {'turma_id': turma_id, 'turma_data': turma_row.to_dict(), 'professor_id': prof.id, 'sala_id': sala.id, 'bloco_inicio': start_slot}
                    timetable.append(assignment)
                    update_schedules(assignment, professor_schedule, room_schedule)
                    scheduled_this_turma = True
                    break # Passa para a próxima turma

        if not scheduled_this_turma:
            unscheduled_turmas_ids.append(turma_id)
            # Adiciona um placeholder para a turma não alocada na representação do AG
            timetable.append({'turma_id': turma_id, 'turma_data': turma_row.to_dict(), 'professor_id': None, 'sala_id': None, 'bloco_inicio': None})
            
    num_scheduled = len(turmas_df) - len(unscheduled_turmas_ids)
    print(f"Heurística construtiva finalizada. Turmas alocadas: {num_scheduled}/{len(turmas_df)}.")
    if unscheduled_turmas_ids: print(f"Turmas não alocadas: {unscheduled_turmas_ids}")
    
    # Para o AG, é útil que o cromossomo sempre tenha um tamanho fixo (número total de turmas)
    # Vamos reordenar a `timetable` para corresponder à ordem original de `data['turmas']`.
    original_turma_order = data['turmas']['id_turma'].tolist()
    ordered_timetable = [None] * len(original_turma_order)
    temp_assignment_map = {entry['turma_id']: entry for entry in timetable}
    for i, turma_id_orig in enumerate(original_turma_order):
        if turma_id_orig in temp_assignment_map: ordered_timetable[i] = temp_assignment_map[turma_id_orig]
        else:
            # Não deve acontecer se a heurística processar todas as turmas
            turma_data_orig = data['turmas'][data['turmas']['id_turma'] == turma_id_orig].iloc[0].to_dict()
            ordered_timetable[i] = {'turma_id': turma_id_orig, 'turma_data': turma_data_orig, 'professor_id': None, 'sala_id': None, 'bloco_inicio': None}
            
    return ordered_timetable # Este é agora um cromossomo para o AG


# --- 4. Algoritmo Genético ---

def get_schedules_from_chromosome(chromosome, data):
    """Reconstrói os horários dos professores e salas a partir de um cromossomo para validação."""
    professor_schedule = defaultdict(set)
    room_schedule = defaultdict(set)
    for _, prof_row in data['professores'].iterrows():
        professor_schedule[prof_row['id']] = prof_row['indisponibilidade_indices'].copy()
    for assignment in chromosome:
        if assignment['professor_id'] is not None:
            update_schedules(assignment, professor_schedule, room_schedule)
    return professor_schedule, room_schedule


def calculate_fitness(chromosome, data, soft_constraint_weights=None):
    """Calcula a aptidão (fitness) de um cromossomo."""
    fitness = 0
    hard_constraint_violations = 0 # Para outras restrições rígidas

    # Para cada aula na grade (cromossomo)
    for i, assignment_i in enumerate(chromosome):
        if assignment_i['professor_id'] is None:
            fitness -= UNSCHEDULED_CLASS_PENALTY # Penalidade por turma não alocada
            continue
        
        turma_i = assignment_i['turma_data']
        prof_i_id = assignment_i['professor_id']
        sala_i_id = assignment_i['sala_id']
        bloco_i_inicio = assignment_i['bloco_inicio']
        duracao_i = turma_i['duracao_blocos']

        prof_i_obj = data['professores'][data['professores']['id'] == prof_i_id].iloc[0]
        sala_i_obj = data['salas'][data['salas']['id'] == sala_i_id].iloc[0]

        # Validações de restrições rígidas
        if not turma_i['recursos_necessarios'].issubset(sala_i_obj['recursos']): hard_constraint_violations += 1
        if turma_i['alunos_estimados'] > sala_i_obj['capacidade']: hard_constraint_violations += 1
            
        # Verificamos cada bloco que a aula ocupa
        for k in range(duracao_i):
            current_slot = bloco_i_inicio + k
            # Se o bloco da aula estiver no conjunto de indisponibilidades fixas do professor...
            if current_slot in prof_i_obj['indisponibilidade_indices']:
                # ...aplicamos diretamente a penalidade alta e específica.
                fitness -= PROFESSOR_UNAVAILABLE_PENALTY
                # Interrompemos o loop interno para penalizar apenas uma vez por alocação inválida.
                break 
    
    # Validação de sobreposição de recursos (professores e salas)
    temp_prof_schedule = defaultdict(set)
    temp_room_schedule = defaultdict(set)
    for assignment in chromosome:
        if assignment['professor_id'] is not None:
            turma = assignment['turma_data']
            prof_id = assignment['professor_id']
            room_id = assignment['sala_id']
            start_slot = assignment['bloco_inicio']
            duration = turma['duracao_blocos']
            for k in range(duration):
                slot = start_slot + k
                if slot in temp_prof_schedule.get(prof_id, set()): hard_constraint_violations += 1; break
                temp_prof_schedule[prof_id].add(slot)
                if slot in temp_room_schedule.get(room_id, set()): hard_constraint_violations += 1; break
                temp_room_schedule[room_id].add(slot)
            if hard_constraint_violations > 0 and assignment == chromosome[-1]: break

    # Aplica a penalidade para as violações rígidas encontradas
    fitness -= hard_constraint_violations * 500

    # Exemplo de Restrição Flexível: preferir horários mais cedo
    total_slot_indices_used = 0
    num_scheduled_classes = sum(1 for a in chromosome if a['professor_id'] is not None)
    if num_scheduled_classes > 0:
        for assignment in chromosome:
            if assignment['bloco_inicio'] is not None:
                total_slot_indices_used += assignment['bloco_inicio']
        fitness -= total_slot_indices_used / num_scheduled_classes

    return fitness


def initialize_population(data, heuristic_solution):
    """Inicializa a população para o AG."""
    population = [heuristic_solution] # Começa com a solução da heurística
    base_prof_schedule, base_room_schedule = get_schedules_from_chromosome(heuristic_solution, data)
    
    # Cria novos indivíduos mutando levemente a solução da heurística para gerar diversidade
    for _ in range(POPULATION_SIZE - 1):
        mutant = copy.deepcopy(heuristic_solution)
        for _i_mut in range(max(1,int(len(mutant)*0.1))):
             mutant = mutate(mutant, data, dict(base_prof_schedule), dict(base_room_schedule))
        population.append(mutant)
    return population

def selection(population, fitness_scores):
    """Seleciona pais usando seleção por torneio."""
    selected_parents = []
    tournament_size = 3
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_contenders = [(fitness_scores[i], population[i]) for i in tournament_indices]
        winner = max(tournament_contenders, key=lambda x: x[0])[1]
        selected_parents.append(winner)
    return selected_parents

def crossover(parent1, parent2, data):
    """Realiza o cruzamento (crossover) de um ponto."""
    if random.random() > CROSSOVER_RATE: return copy.deepcopy(parent1), copy.deepcopy(parent2) # Sem cruzamento
    point = random.randint(1, len(parent1) - 2)
    
    offspring1_assignments = parent1[:point] + parent2[point:]
    offspring2_assignments = parent2[:point] + parent1[point:]
    return offspring1_assignments, offspring2_assignments

def mutate(individual, data, current_prof_schedule, current_room_schedule):
    """Realiza mutação em um indivíduo."""
    mutated_individual = copy.deepcopy(individual)
    if random.random() > MUTATION_RATE: return mutated_individual
    
    class_idx_to_mutate = random.randrange(len(mutated_individual))
    current_assignment = mutated_individual[class_idx_to_mutate]
    turma_to_mutate = current_assignment['turma_data']
    
    if current_assignment['professor_id'] is not None:
        clear_schedules_for_class(turma_to_mutate, current_assignment, current_prof_schedule, current_room_schedule)
        
    found_new_assignment = False
    potential_professors = [p for p_id in turma_to_mutate['prof_compativel_ids'] for p in [data['professores'][data['professores']['id'] == p_id].iloc[0]] if not p.empty]
    random.shuffle(potential_professors)
    
    potential_salas_df = data['salas'][(data['salas']['recursos'].apply(lambda r: turma_to_mutate['recursos_necessarios'].issubset(r))) & (data['salas']['capacidade'] >= turma_to_mutate['alunos_estimados'])]
    potential_salas = list(potential_salas_df.itertuples())
    random.shuffle(potential_salas)
    
    shuffled_time_slots = list(range(data['total_time_slots']))
    random.shuffle(shuffled_time_slots)
    
    for prof_obj in potential_professors[:5]:
        for sala_obj in potential_salas[:5]:
            for start_slot in shuffled_time_slots[:20]:
                if not is_slot_sequence_valid(start_slot, turma_to_mutate['duracao_blocos'], data['total_time_slots']): continue
                prof_available = check_professor_availability(prof_obj.id, start_slot, turma_to_mutate['duracao_blocos'], current_prof_schedule, prof_obj['indisponibilidade_indices'])
                if not prof_available: continue
                room_available = check_room_availability(sala_obj.id, start_slot, turma_to_mutate['duracao_blocos'], current_room_schedule)
                if not room_available: continue
                new_assignment = {'turma_id': turma_to_mutate['id_turma'], 'turma_data': turma_to_mutate, 'professor_id': prof_obj.id, 'sala_id': sala_obj.id, 'bloco_inicio': start_slot}
                mutated_individual[class_idx_to_mutate] = new_assignment
                update_schedules(new_assignment, current_prof_schedule, current_room_schedule)
                found_new_assignment = True
                break
            if found_new_assignment: break
        if found_new_assignment: break
        
    if not found_new_assignment:
         mutated_individual[class_idx_to_mutate] = {'turma_id': turma_to_mutate['id_turma'], 'turma_data': turma_to_mutate, 'professor_id': None, 'sala_id': None, 'bloco_inicio': None}
    
    return mutated_individual

def genetic_algorithm(data, initial_solution):
    """Executa o Algoritmo Genético para encontrar uma grade horária otimizada."""
    print("\nIniciando o Algoritmo Genético...")
    population = initialize_population(data, initial_solution)
    best_solution_overall = initial_solution
    best_fitness_overall = calculate_fitness(initial_solution, data)
    print(f"Fitness inicial (da heurística): {best_fitness_overall:.2f}")
    
    for generation in range(MAX_GENERATIONS):
        fitness_scores = [calculate_fitness(ind, data) for ind in population]
        new_population = []
        
        # Elitismo: Mantém os melhores indivíduos
        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        for i in range(ELITISM_COUNT): new_population.append(copy.deepcopy(sorted_population[i][0]))
        
        current_best_fitness_gen = sorted_population[0][1]
        if current_best_fitness_gen > best_fitness_overall:
            best_fitness_overall = current_best_fitness_gen
            best_solution_overall = copy.deepcopy(sorted_population[0][0])
            
        print(f"Geração {generation+1}/{MAX_GENERATIONS} - Melhor Fitness: {best_fitness_overall:.2f}")
        
        # Seleção, Cruzamento, Mutação
        selected_parents = selection(population, fitness_scores)
        
        # Itera em pares para o cruzamento
        for i in range(0, len(selected_parents) - ELITISM_COUNT, 2):
            if i + 1 >= len(selected_parents):
                if selected_parents: new_population.append(copy.deepcopy(selected_parents[i]))
                break
            
            parent1 = selected_parents[i]
            parent2 = selected_parents[i+1]
            offspring1, offspring2 = crossover(parent1, parent2, data)
            
            temp_prof_schedule1, temp_room_schedule1 = get_schedules_from_chromosome(offspring1, data)
            offspring1_mutated = mutate(offspring1, data, temp_prof_schedule1, temp_room_schedule1)
            
            temp_prof_schedule2, temp_room_schedule2 = get_schedules_from_chromosome(offspring2, data)
            offspring2_mutated = mutate(offspring2, data, temp_prof_schedule2, temp_room_schedule2)
            
            new_population.append(offspring1_mutated)
            if len(new_population) < POPULATION_SIZE: new_population.append(offspring2_mutated)
            
        population = new_population[:POPULATION_SIZE] # Garante que o tamanho da população seja mantido
        
    return best_solution_overall, best_fitness_overall


# --- 5. Execução Principal ---
if __name__ == '__main__':
    data = load_and_preprocess_data()

    if data:
        # Executa a Heurística Construtiva
        initial_timetable_chromosome = constructive_heuristic(data)
        
        # Executa o Algoritmo Genético
        final_timetable, final_fitness = genetic_algorithm(data, initial_timetable_chromosome)

        # Nome do arquivo de saída
        output_filename = "meta_heuristicas\projeto_final\saida\grade_horaria_final.txt"

        # Escreve a saída principal no arquivo
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write("--- Final Timetable (Best Solution from GA) ---\n\n")
                num_actually_scheduled = 0
                if final_timetable:
                    for assignment in final_timetable:
                        if assignment['professor_id']:
                            num_actually_scheduled += 1
                            turma = assignment['turma_data']
                            prof_name = data['professores'][data['professores']['id'] == assignment['professor_id']]['nome'].iloc[0]
                            sala_name = data['salas'][data['salas']['id'] == assignment['sala_id']]['nome'].iloc[0]
                            time_str = data['time_slots_map'][assignment['bloco_inicio']]
                            
                            f.write(f"Turma: {assignment['turma_id']} ({turma['nome']})\n")
                            f.write(f"  Professor: {prof_name} ({assignment['professor_id']})\n")
                            f.write(f"  Sala: {sala_name} ({assignment['sala_id']})\n")
                            f.write(f"  Horario: Slot {assignment['bloco_inicio']} ({time_str}) for {turma['duracao_blocos']} blocos\n")
                            f.write("-" * 20 + "\n")
                    
                    f.write(f"\nTotal turmas scheduled in final solution: {num_actually_scheduled} / {len(data['turmas'])}\n")
                    f.write(f"Final Fitness: {final_fitness:.2f}\n")
                else:
                    f.write("No solution found.\n")
            
            print(f"\nSaída principal do programa foi escrita no arquivo: {output_filename}")
        except IOError:
            print(f"Erro: Não foi possível escrever no arquivo {output_filename}")

# 2. Grava o tempo de fim
end_time = time.time()

# 3. Calcula a diferença e imprime o tempo total de execução
execution_time = end_time - start_time
print(f"\nO tempo total de execução foi de: {execution_time} segundos")