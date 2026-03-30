"""
Serviço Report Agent
Geração de relatórios de simulação no padrão ReACT usando LangChain + Zep

Funcionalidades:
1. Gerar relatórios com base nos requisitos de simulação e nas informações do grafo Zep
2. Planejar a estrutura do sumário primeiro, depois gerar por seções
3. Cada seção utiliza o modo ReACT com múltiplas rodadas de raciocínio e reflexão
4. Suporte a diálogo com o usuário, invocando ferramentas de busca autonomamente durante a conversa
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Registrador de logs detalhados do Report Agent
    
    Gera o arquivo agent_log.jsonl na pasta do relatório, registrando cada ação detalhada.
    Cada linha é um objeto JSON completo, contendo timestamp, tipo de ação, conteúdo detalhado, etc.
    """
    
    def __init__(self, report_id: str):
        """
        Inicializar o registrador de logs
        
        Args:
            report_id: ID do relatório, usado para determinar o caminho do arquivo de log
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Garantir que o diretório do arquivo de log exista"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Obter o tempo decorrido desde o início (em segundos)"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Registrar uma entrada de log
        
        Args:
            action: Tipo de ação, ex: 'start', 'tool_call', 'llm_response', 'section_complete' etc.
            stage: Estágio atual, ex: 'planning', 'generating', 'completed'
            details: Dicionário de conteúdo detalhado, sem truncamento
            section_title: Título da seção atual (opcional)
            section_index: Índice da seção atual (opcional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Anexar ao arquivo JSONL
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Registrar o início da geração do relatório"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Tarefa de geração de relatório iniciada"
            }
        )
    
    def log_planning_start(self):
        """Registrar o início do planejamento do esboço"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Iniciando planejamento do esboço do relatório"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Registrar as informações de contexto obtidas durante o planejamento"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Informações de contexto da simulação obtidas",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Registrar a conclusão do planejamento do esboço"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Planejamento do esboço concluído",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Registrar o início da geração de uma seção"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Iniciando geração da seção: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Registrar o processo de raciocínio ReACT"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT rodada {iteration} de raciocínio"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Registrar chamada de ferramenta"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Chamando ferramenta: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Registrar resultado da chamada de ferramenta (conteúdo completo, sem truncamento)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Resultado completo, sem truncamento
                "result_length": len(result),
                "message": f"Ferramenta {tool_name} retornou resultado"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Registrar resposta do LLM (conteúdo completo, sem truncamento)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Resposta completa, sem truncamento
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"Resposta do LLM (chamada de ferramenta: {has_tool_calls}, resposta final: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Registrar conclusão da geração de conteúdo da seção (apenas registra o conteúdo, não significa que a seção inteira está concluída)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Conteúdo completo, sem truncamento
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Conteúdo da seção {section_title} gerado com sucesso"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Registrar conclusão da geração da seção

        O frontend deve monitorar este log para determinar se uma seção foi realmente concluída e obter o conteúdo completo
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Seção {section_title} gerada com sucesso"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Registrar conclusão da geração do relatório"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Geração do relatório concluída"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Registrar erro"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Erro ocorrido: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Registrador de logs de console do Report Agent
    
    Grava logs no estilo console (INFO, WARNING, etc.) no arquivo console_log.txt dentro da pasta do relatório.
    Esses logs são diferentes do agent_log.jsonl — são saídas de console em formato texto puro.
    """
    
    def __init__(self, report_id: str):
        """
        Inicializar o registrador de logs de console
        
        Args:
            report_id: ID do relatório, usado para determinar o caminho do arquivo de log
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Garantir que o diretório do arquivo de log exista"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Configurar o manipulador de arquivo para gravar logs simultaneamente no arquivo"""
        import logging
        
        # Criar manipulador de arquivo
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Usar o mesmo formato conciso do console
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Adicionar aos loggers relacionados ao report_agent
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Evitar adição duplicada
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Fechar o manipulador de arquivo e removê-lo do logger"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Garantir que o manipulador de arquivo seja fechado durante a destruição"""
        self.close()


class ReportStatus(str, Enum):
    """Status do relatório"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Seção do relatório"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Converter para formato Markdown"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Esboço do relatório"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Converter para formato Markdown"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Relatório completo"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Constantes de templates de prompt
# ═══════════════════════════════════════════════════════════════

# ── Descrição das ferramentas ──

TOOL_DESC_INSIGHT_FORGE = """\
【Recuperação de Insights Profundos - Ferramenta de busca poderosa】
Esta é nossa poderosa função de recuperação, projetada para análise profunda. Ela irá:
1. Decompor automaticamente sua pergunta em múltiplas subquestões
2. Buscar informações no grafo de simulação a partir de múltiplas dimensões
3. Integrar resultados de busca semântica, análise de entidades e rastreamento de cadeias de relações
4. Retornar o conteúdo de recuperação mais abrangente e profundo

【Cenários de uso】
- Necessidade de análise aprofundada sobre um tema
- Necessidade de compreender múltiplos aspectos de um evento
- Necessidade de obter material rico para fundamentar seções do relatório

【Conteúdo retornado】
- Textos originais de fatos relevantes (citáveis diretamente)
- Insights sobre entidades principais
- Análise de cadeias de relações"""

TOOL_DESC_PANORAMA_SEARCH = """\
【Busca ampla - Obter visão panorâmica】
Esta ferramenta é usada para obter uma visão panorâmica completa dos resultados da simulação, especialmente adequada para entender a evolução dos eventos. Ela irá:
1. Obter todos os nós e relações relevantes
2. Distinguir entre fatos atualmente válidos e fatos históricos/expirados
3. Ajudá-lo a entender como a opinião pública evoluiu

【Cenários de uso】
- Necessidade de compreender a trajetória completa de desenvolvimento de um evento
- Necessidade de comparar mudanças na opinião pública em diferentes fases
- Necessidade de obter informações completas sobre entidades e relações

【Conteúdo retornado】
- Fatos atualmente válidos (resultados mais recentes da simulação)
- Fatos históricos/expirados (registros de evolução)
- Todas as entidades envolvidas"""

TOOL_DESC_QUICK_SEARCH = """\
【Busca simples - Recuperação rápida】
Ferramenta de recuperação rápida e leve, adequada para consultas simples e diretas.

【Cenários de uso】
- Necessidade de buscar rapidamente uma informação específica
- Necessidade de verificar um fato
- Recuperação simples de informações

【Conteúdo retornado】
- Lista de fatos mais relevantes para a consulta"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【Entrevista profunda - Entrevista real com Agents (duas plataformas)】
Chama a API de entrevista do ambiente de simulação OASIS para realizar entrevistas reais com os Agents em execução!
Isto não é simulação de LLM, mas sim chamadas à interface real de entrevista para obter respostas originais dos Agents.
Por padrão, entrevista simultaneamente nas plataformas Twitter e Reddit para obter perspectivas mais abrangentes.

Fluxo funcional:
1. Lê automaticamente o arquivo de perfis para conhecer todos os Agents da simulação
2. Seleciona inteligentemente os Agents mais relevantes para o tema da entrevista (ex: estudantes, mídia, autoridades oficiais, etc.)
3. Gera automaticamente perguntas para a entrevista
4. Chama a interface /api/simulation/interview/batch para realizar entrevistas reais em ambas as plataformas
5. Integra todos os resultados das entrevistas, fornecendo análise multi-perspectiva

【Cenários de uso】
- Necessidade de compreender a visão do evento a partir de diferentes papéis (o que os estudantes pensam? O que a mídia acha? O que as autoridades dizem?)
- Necessidade de coletar opiniões e posições de múltiplas partes
- Necessidade de obter respostas reais dos Agents da simulação (do ambiente de simulação OASIS)
- Desejo de tornar o relatório mais vivo, incluindo "registros de entrevistas"

【Conteúdo retornado】
- Informações de identidade dos Agents entrevistados
- Respostas das entrevistas de cada Agent nas plataformas Twitter e Reddit
- Citações-chave (citáveis diretamente)
- Resumo da entrevista e comparação de pontos de vista

【Importante】O ambiente de simulação OASIS precisa estar em execução para usar esta funcionalidade!"""

# ── Prompt de planejamento do esboço ──

PLAN_SYSTEM_PROMPT = """\
Você é um especialista em redação de «Relatórios de Predição Futura», com uma «visão de Deus» sobre o mundo simulado — você pode observar o comportamento, as falas e as interações de cada Agent na simulação.

【Conceito central】
Nós construímos um mundo simulado e injetamos um «requisito de simulação» específico como variável. O resultado da evolução do mundo simulado é a predição do que pode acontecer no futuro. O que você está observando não são "dados experimentais", mas sim um "ensaio do futuro".

【Sua tarefa】
Redigir um «Relatório de Predição Futura» que responda:
1. Sob as condições que definimos, o que aconteceu no futuro?
2. Como os diferentes tipos de Agents (grupos de pessoas) reagiram e agiram?
3. Quais tendências e riscos futuros dignos de atenção esta simulação revelou?

【Posicionamento do relatório】
- ✅ Este é um relatório de predição futura baseado em simulação, revelando "se isso acontecer, como será o futuro"
- ✅ Foco nos resultados preditivos: direção dos eventos, reações dos grupos, fenômenos emergentes, riscos potenciais
- ✅ As falas e ações dos Agents no mundo simulado são predições do comportamento futuro das pessoas
- ❌ Não é uma análise da situação atual do mundo real
- ❌ Não é uma revisão genérica de opinião pública

【Limite de número de seções】
- Mínimo de 2 seções, máximo de 5 seções
- Não são necessárias subseções; cada seção contém o conteúdo completo diretamente
- O conteúdo deve ser conciso, focado nas descobertas preditivas essenciais
- A estrutura das seções é projetada por você com base nos resultados da predição

Por favor, produza o esboço do relatório em formato JSON, conforme abaixo:
{
    "title": "Título do relatório",
    "summary": "Resumo do relatório (uma frase sintetizando a principal descoberta preditiva)",
    "sections": [
        {
            "title": "Título da seção",
            "description": "Descrição do conteúdo da seção"
        }
    ]
}

Atenção: o array sections deve ter no mínimo 2 e no máximo 5 elementos!"""

PLAN_USER_PROMPT_TEMPLATE = """\
【Configuração do cenário de predição】
Variável injetada no mundo simulado (requisito de simulação): {simulation_requirement}

【Escala do mundo simulado】
- Número de entidades participantes na simulação: {total_nodes}
- Número de relações geradas entre entidades: {total_edges}
- Distribuição por tipo de entidade: {entity_types}
- Número de Agents ativos: {total_entities}

【Amostra de fatos futuros previstos pela simulação】
{related_facts_json}

Observe este ensaio do futuro com a «visão de Deus»:
1. Sob as condições que definimos, que estado o futuro apresentou?
2. Como os diferentes tipos de pessoas (Agents) reagiram e agiram?
3. Quais tendências futuras dignos de atenção esta simulação revelou?

Com base nos resultados da predição, projete a estrutura de seções mais adequada para o relatório.

【Lembrete】Número de seções do relatório: mínimo 2, máximo 5 — o conteúdo deve ser conciso e focado nas descobertas preditivas essenciais."""

# ── Prompt de geração de seção ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
Você é um especialista em redação de «Relatórios de Predição Futura» e está redigindo uma seção do relatório.

Título do relatório: {report_title}
Resumo do relatório: {report_summary}
Cenário de predição (requisito de simulação): {simulation_requirement}

Seção a ser redigida: {section_title}

═══════════════════════════════════════════════════════════════
【Conceito central】
═══════════════════════════════════════════════════════════════

O mundo simulado é um ensaio do futuro. Nós injetamos condições específicas (requisito de simulação) no mundo simulado,
e o comportamento e as interações dos Agents na simulação são predições do comportamento futuro das pessoas.

Sua tarefa é:
- Revelar o que aconteceu no futuro sob as condições definidas
- Prever como os diferentes tipos de pessoas (Agents) reagiram e agiram
- Descobrir tendências futuras, riscos e oportunidades dignos de atenção

❌ Não escreva como uma análise da situação atual do mundo real
✅ Foque em "como será o futuro" — os resultados da simulação são o futuro previsto

═══════════════════════════════════════════════════════════════
【Regras mais importantes - OBRIGATÓRIAS】
═══════════════════════════════════════════════════════════════

1. 【É OBRIGATÓRIO chamar ferramentas para observar o mundo simulado】
   - Você está observando o ensaio do futuro com a «visão de Deus»
   - Todo o conteúdo deve vir dos eventos e falas dos Agents no mundo simulado
   - É proibido usar seu próprio conhecimento para escrever o conteúdo do relatório
   - Cada seção deve chamar ferramentas pelo menos 3 vezes (máximo 5) para observar o mundo simulado, que representa o futuro

2. 【É OBRIGATÓRIO citar falas e ações originais dos Agents】
   - As falas e ações dos Agents são predições do comportamento futuro das pessoas
   - Use o formato de citação no relatório para exibir essas predições, por exemplo:
     > "Um certo grupo de pessoas expressou: conteúdo original..."
   - Essas citações são as evidências centrais da predição da simulação

3. 【Consistência linguística - conteúdo citado deve ser traduzido para o idioma do relatório】
   - O conteúdo retornado pelas ferramentas pode conter expressões em inglês ou mistas
   - Se o requisito de simulação e o material original estão em português, o relatório deve ser inteiramente em português
   - Ao citar conteúdo em inglês ou misto retornado pelas ferramentas, traduza para português fluente antes de incluir no relatório
   - Mantenha o significado original ao traduzir, garantindo expressão natural e fluente
   - Esta regra se aplica tanto ao texto corrido quanto ao conteúdo em blocos de citação (formato >)

4. 【Apresentar fielmente os resultados da predição】
   - O conteúdo do relatório deve refletir os resultados da simulação que representam o futuro
   - Não adicione informações que não existem na simulação
   - Se as informações sobre algum aspecto forem insuficientes, declare isso honestamente

═══════════════════════════════════════════════════════════════
【⚠️ Normas de formatação - EXTREMAMENTE IMPORTANTE!】
═══════════════════════════════════════════════════════════════

【Uma seção = unidade mínima de conteúdo】
- Cada seção é a menor unidade de divisão do relatório
- ❌ Proibido usar qualquer título Markdown (#, ##, ###, #### etc.) dentro da seção
- ❌ Proibido adicionar o título principal da seção no início do conteúdo
- ✅ O título da seção é adicionado automaticamente pelo sistema; você só precisa escrever o texto corrido
- ✅ Use **negrito**, separação de parágrafos, citações e listas para organizar o conteúdo, mas não use títulos

【Exemplo correto】
```
Esta seção analisa a dinâmica de propagação da opinião pública do evento. Através de uma análise aprofundada dos dados de simulação, descobrimos...

**Fase de explosão inicial**

O Weibo, como a primeira cena da opinião pública, desempenhou a função central de divulgação inicial:

> "O Weibo contribuiu com 68% do volume de voz inicial..."

**Fase de amplificação emocional**

A plataforma Douyin amplificou ainda mais o impacto do evento:

- Alto impacto visual
- Alta ressonância emocional
```

【Exemplo incorreto】
```
## Resumo executivo          ← Errado! Não adicione nenhum título
### 1. Fase inicial     ← Errado! Não use ### para subseções
#### 1.1 Análise detalhada   ← Errado! Não use #### para subdivisões

Esta seção analisou...
```

═══════════════════════════════════════════════════════════════
【Ferramentas de recuperação disponíveis】 (chamar 3-5 vezes por seção)
═══════════════════════════════════════════════════════════════

{tools_description}

【Sugestões de uso de ferramentas - use ferramentas variadas, não use apenas uma】
- insight_forge: Análise de insights profundos, decompõe problemas automaticamente e busca fatos e relações em múltiplas dimensões
- panorama_search: Busca panorâmica ampla, para entender o panorama completo do evento, linha do tempo e evolução
- quick_search: Verificação rápida de um ponto de informação específico
- interview_agents: Entrevistar Agents da simulação, obter pontos de vista em primeira pessoa e reações reais de diferentes papéis

═══════════════════════════════════════════════════════════════
【Fluxo de trabalho】
═══════════════════════════════════════════════════════════════

Em cada resposta você só pode fazer uma das duas coisas a seguir (não ambas ao mesmo tempo):

Opção A - Chamar ferramenta:
Exponha seu raciocínio, depois chame uma ferramenta no seguinte formato:
<tool_call>
{{"name": "nome_da_ferramenta", "parameters": {{"nome_parametro": "valor_parametro"}}}}
</tool_call>
O sistema executará a ferramenta e retornará o resultado para você. Você não precisa e não pode escrever o resultado da ferramenta por conta própria.

Opção B - Gerar conteúdo final:
Quando você já obteve informações suficientes via ferramentas, comece com "Final Answer:" e produza o conteúdo da seção.

⚠️ Estritamente proibido:
- Proibido incluir chamada de ferramenta e Final Answer na mesma resposta
- Proibido inventar resultados de ferramentas (Observation); todos os resultados são injetados pelo sistema
- No máximo uma chamada de ferramenta por resposta

═══════════════════════════════════════════════════════════════
【Requisitos de conteúdo da seção】
═══════════════════════════════════════════════════════════════

1. O conteúdo deve ser baseado nos dados de simulação obtidos pelas ferramentas
2. Cite abundantemente textos originais para demonstrar os resultados da simulação
3. Use formato Markdown (mas proibido usar títulos):
   - Use **texto em negrito** para destacar pontos-chave (substituindo subtítulos)
   - Use listas (- ou 1.2.3.) para organizar pontos
   - Use linhas em branco para separar parágrafos diferentes
   - ❌ Proibido usar #, ##, ###, #### ou qualquer sintaxe de título
4. 【Norma de formato de citação - deve ser parágrafo independente】
   Citações devem ser parágrafos independentes, com uma linha em branco antes e depois, não misturadas dentro de parágrafos:

   ✅ Formato correto:
   ```
   A resposta da escola foi considerada sem conteúdo substancial.

   > "O modelo de resposta da escola parece rígido e lento no ambiente dinâmico das mídias sociais."

   Esta avaliação reflete a insatisfação geral do público.
   ```

   ❌ Formato incorreto:
   ```
   A resposta da escola foi considerada sem conteúdo substancial.> "O modelo de resposta da escola..." Esta avaliação reflete...
   ```
5. Manter coerência lógica com as outras seções
6. 【Evitar repetições】Leia atentamente o conteúdo das seções já concluídas abaixo; não repita as mesmas informações
7. 【Reforçando】Não adicione nenhum título! Use **negrito** para substituir subtítulos"""

SECTION_USER_PROMPT_TEMPLATE = """\
Conteúdo das seções já concluídas (leia atentamente para evitar repetições):
{previous_content}

═══════════════════════════════════════════════════════════════
【Tarefa atual】Redigir seção: {section_title}
═══════════════════════════════════════════════════════════════

【Lembretes importantes】
1. Leia atentamente as seções já concluídas acima, evite repetir o mesmo conteúdo!
2. Antes de começar, é obrigatório chamar ferramentas para obter dados da simulação
3. Use ferramentas variadas, não use apenas uma
4. O conteúdo do relatório deve vir dos resultados da busca, não use seu próprio conhecimento

【⚠️ Aviso de formatação - OBRIGATÓRIO】
- ❌ Não escreva nenhum título (#, ##, ###, #### são todos proibidos)
- ❌ Não escreva "{section_title}" como início
- ✅ O título da seção é adicionado automaticamente pelo sistema
- ✅ Escreva diretamente o texto, use **negrito** para substituir subtítulos

Comece:
1. Primeiro pense (Thought) sobre que informações esta seção precisa
2. Depois chame ferramentas (Action) para obter dados da simulação
3. Após coletar informações suficientes, produza o Final Answer (texto puro, sem nenhum título)"""

# ── Templates de mensagens do ciclo ReACT ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (resultado da busca):

═══ Ferramenta {tool_name} retornou ═══
{result}

═══════════════════════════════════════════════════════════════
Ferramentas chamadas {tool_calls_count}/{max_tool_calls} vezes (usadas: {used_tools_str}){unused_hint}
- Se as informações forem suficientes: comece com "Final Answer:" e produza o conteúdo da seção (deve citar os textos originais acima)
- Se precisar de mais informações: chame uma ferramenta para continuar a busca
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【Atenção】Você chamou ferramentas apenas {tool_calls_count} vezes; são necessárias pelo menos {min_tool_calls} vezes."
    "Chame mais ferramentas para obter mais dados de simulação antes de produzir o Final Answer.{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Atualmente foram chamadas apenas {tool_calls_count} ferramentas; são necessárias pelo menos {min_tool_calls}."
    "Chame ferramentas para obter dados da simulação.{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "O número de chamadas de ferramentas atingiu o limite ({tool_calls_count}/{max_tool_calls}); não é mais possível chamar ferramentas."
    'Com base nas informações já obtidas, produza imediatamente o conteúdo da seção começando com "Final Answer:".'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 Você ainda não usou: {unused_list}; é recomendado experimentar diferentes ferramentas para obter informações de múltiplos ângulos"

REACT_FORCE_FINAL_MSG = "O limite de chamadas de ferramentas foi atingido. Produza diretamente o Final Answer: e gere o conteúdo da seção."

# ── Prompt de Chat ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
Você é um assistente de predição por simulação, conciso e eficiente.

【Contexto】
Condições de predição: {simulation_requirement}

【Relatório de análise já gerado】
{report_content}

【Regras】
1. Priorize respostas baseadas no conteúdo do relatório acima
2. Responda diretamente à pergunta, evite longas dissertações
3. Somente chame ferramentas quando o conteúdo do relatório for insuficiente para responder
4. As respostas devem ser concisas, claras e organizadas

【Ferramentas disponíveis】 (use apenas quando necessário, no máximo 1-2 vezes)
{tools_description}

【Formato de chamada de ferramenta】
<tool_call>
{{"name": "nome_da_ferramenta", "parameters": {{"nome_parametro": "valor_parametro"}}}}
</tool_call>

【Estilo de resposta】
- Conciso e direto, sem textos longos
- Use formato > para citar conteúdo-chave
- Apresente a conclusão primeiro, depois explique o motivo"""

CHAT_OBSERVATION_SUFFIX = "\n\nResponda à pergunta de forma concisa."


# ═══════════════════════════════════════════════════════════════
# Classe principal ReportAgent
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - Agent de geração de relatórios de simulação

    Utiliza o padrão ReACT (Reasoning + Acting):
    1. Fase de planejamento: analisar requisitos de simulação, planejar estrutura do sumário
    2. Fase de geração: gerar conteúdo seção por seção, cada seção pode chamar ferramentas múltiplas vezes
    3. Fase de reflexão: verificar completude e precisão do conteúdo
    """
    
    # Número máximo de chamadas de ferramentas (por seção)
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Número máximo de rodadas de reflexão
    MAX_REFLECTION_ROUNDS = 3
    
    # Número máximo de chamadas de ferramentas no diálogo
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Inicializar Report Agent
        
        Args:
            graph_id: ID do grafo
            simulation_id: ID da simulação
            simulation_requirement: Descrição do requisito de simulação
            llm_client: Cliente LLM (opcional)
            zep_tools: Serviço de ferramentas Zep (opcional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Definição das ferramentas
        self.tools = self._define_tools()
        
        # Registrador de logs (inicializado em generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Registrador de logs de console (inicializado em generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(f"ReportAgent inicializado: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Definir ferramentas disponíveis"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "A questão ou tema que você deseja analisar em profundidade",
                    "report_context": "Contexto da seção atual do relatório (opcional, ajuda a gerar subquestões mais precisas)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Consulta de busca, para ordenação por relevância",
                    "include_expired": "Se deve incluir conteúdo expirado/histórico (padrão True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "String de consulta de busca",
                    "limit": "Número de resultados retornados (opcional, padrão 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Tema ou descrição da necessidade da entrevista (ex: 'entender a opinião dos estudantes sobre o incidente de formaldeído no dormitório')",
                    "max_agents": "Número máximo de Agents a entrevistar (opcional, padrão 5, máximo 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Executar chamada de ferramenta
        
        Args:
            tool_name: Nome da ferramenta
            parameters: Parâmetros da ferramenta
            report_context: Contexto do relatório (para InsightForge)
            
        Returns:
            Resultado da execução da ferramenta (formato texto)
        """
        logger.info(f"Executando ferramenta: {tool_name}, parâmetros: {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Busca ampla - obter panorama completo
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Busca simples - recuperação rápida
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Entrevista profunda - chamar a API de entrevista real do OASIS para obter respostas dos Agents (duas plataformas)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Ferramentas antigas compatíveis (redirecionamento interno para novas ferramentas) ==========
            
            elif tool_name == "search_graph":
                # Redirecionado para quick_search
                logger.info("search_graph redirecionado para quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirecionado para insight_forge, pois é mais poderoso
                logger.info("get_simulation_context redirecionado para insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Ferramenta desconhecida: {tool_name}. Use uma das seguintes ferramentas: insight_forge, panorama_search, quick_search"
                
        except Exception as e:
            logger.error(f"Execução da ferramenta falhou: {tool_name}, erro: {str(e)}")
            return f"Execução da ferramenta falhou: {str(e)}"
    
    # Conjunto de nomes de ferramentas válidos, usado para análise de JSON puro como fallback
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Analisar chamadas de ferramentas a partir da resposta do LLM

        Formatos suportados (por prioridade):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. JSON puro (a resposta inteira ou uma única linha é um JSON de chamada de ferramenta)
        """
        tool_calls = []

        # Formato 1: Estilo XML (formato padrão)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Formato 2: Fallback - LLM gera JSON puro (sem tags <tool_call>)
        # Tentado apenas quando o formato 1 não corresponde, para evitar falsos positivos em JSON do corpo do texto
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # A resposta pode conter texto de raciocínio + JSON puro; tenta extrair o último objeto JSON
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validar se o JSON extraído é uma chamada de ferramenta válida"""
        # Suporta dois formatos de chaves: {"name": ..., "parameters": ...} e {"tool": ..., "params": ...}
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalizar chaves para name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Gerar texto de descrição das ferramentas"""
        desc_parts = ["Ferramentas disponíveis:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parâmetros: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Planejar esboço do relatório
        
        Usar LLM para analisar requisitos de simulação e planejar a estrutura do sumário
        
        Args:
            progress_callback: Função de callback de progresso
            
        Returns:
            ReportOutline: Esboço do relatório
        """
        logger.info("Iniciando planejamento do esboço do relatório...")
        
        if progress_callback:
            progress_callback("planning", 0, "Analisando requisitos de simulação...")
        
        # Primeiro obter o contexto de simulação
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "Gerando esboço do relatório...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "Analisando estrutura do esboço...")
            
            # Analisar esboço
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Relatório de Análise de Simulação"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "Planejamento do esboço concluído")
            
            logger.info(f"Planejamento do esboço concluído: {len(sections)} seções")
            return outline
            
        except Exception as e:
            logger.error(f"Falha no planejamento do esboço: {str(e)}")
            # Retornar esboço padrão (3 seções, como fallback)
            return ReportOutline(
                title="Relatório de Predição Futura",
                summary="Análise de tendências futuras e riscos baseada em simulação preditiva",
                sections=[
                    ReportSection(title="Cenário de Predição e Descobertas Essenciais"),
                    ReportSection(title="Análise Preditiva do Comportamento de Grupos"),
                    ReportSection(title="Perspectivas de Tendências e Alertas de Risco")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Gerar conteúdo de uma seção individual usando o padrão ReACT
        
        Ciclo ReACT:
        1. Thought (Pensamento) - Analisar quais informações são necessárias
        2. Action (Ação) - Chamar ferramentas para obter informações
        3. Observation (Observação) - Analisar resultados retornados pelas ferramentas
        4. Repetir até ter informações suficientes ou atingir o número máximo
        5. Final Answer (Resposta Final) - Gerar conteúdo da seção
        
        Args:
            section: Seção a ser gerada
            outline: Esboço completo
            previous_sections: Conteúdo das seções anteriores (para manter coerência)
            progress_callback: Callback de progresso
            section_index: Índice da seção (para registro de logs)
            
        Returns:
            Conteúdo da seção (formato Markdown)
        """
        logger.info(f"ReACT gerando seção: {section.title}")
        
        # Registrar log de início da seção
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Construir prompt do usuário - cada seção concluída entra com no máximo 4000 caracteres
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # No máximo 4000 caracteres por seção
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(Esta é a primeira seção)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Ciclo ReACT
        tool_calls_count = 0
        max_iterations = 5  # Número máximo de iterações
        min_tool_calls = 3  # Número mínimo de chamadas de ferramentas
        conflict_retries = 0  # Número de conflitos consecutivos com chamada de ferramenta e Final Answer simultâneos
        used_tools = set()  # Registrar ferramentas já chamadas
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Contexto do relatório, usado para geração de subquestões do InsightForge
        report_context = f"Título da seção: {section.title}\nRequisito de simulação: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Recuperação profunda e redação em andamento ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # Chamar LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Verificar se o retorno do LLM é None (exceção de API ou conteúdo vazio)
            if response is None:
                logger.warning(f"Seção {section.title} iteração {iteration + 1}: LLM retornou None")
                # Se ainda há iterações, adicionar mensagem e tentar novamente
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(resposta vazia)"})
                    messages.append({"role": "user", "content": "Por favor, continue gerando o conteúdo."})
                    continue
                # Última iteração também retornou None, sair do loop para finalização forçada
                break

            logger.debug(f"Resposta do LLM: {response[:200]}...")

            # Analisar uma vez e reutilizar o resultado
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Tratamento de conflito: LLM gerou chamada de ferramenta e Final Answer simultaneamente ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Seção {section.title} rodada {iteration+1}: "
                    f"LLM gerou chamada de ferramenta e Final Answer simultaneamente (conflito nº {conflict_retries})"
                )

                if conflict_retries <= 2:
                    # Primeiras duas vezes: descartar esta resposta e pedir ao LLM para responder novamente
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[❗Erro de formato] Você incluiu chamada de ferramenta e Final Answer na mesma resposta, o que não é permitido.\n"
                            "Cada resposta só pode fazer uma das duas coisas a seguir:\n"
                            "- Chamar uma ferramenta (gerar um bloco <tool_call>, sem escrever Final Answer)\n"
                            "- Gerar conteúdo final (começar com 'Final Answer:', sem incluir <tool_call>)\n"
                            "Por favor, responda novamente, fazendo apenas uma das opções."
                        ),
                    })
                    continue
                else:
                    # Terceira vez: tratamento degradado, truncar até a primeira chamada de ferramenta e executar forçadamente
                    logger.warning(
                        f"Seção {section.title}: {conflict_retries} conflitos consecutivos, "
                        "degradando para truncar e executar a primeira chamada de ferramenta"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Registrar log da resposta do LLM
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Situação 1: LLM gerou Final Answer ──
            if has_final_answer:
                # Número insuficiente de chamadas de ferramentas, rejeitar e solicitar mais chamadas
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(Estas ferramentas ainda não foram usadas, recomenda-se experimentá-las: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Término normal
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Seção {section.title} geração concluída (chamadas de ferramenta: {tool_calls_count} vezes)")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Situação 2: LLM tentou chamar ferramenta ──
            if has_tool_calls:
                # Cota de ferramentas esgotada → informar claramente e solicitar Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Executar apenas a primeira chamada de ferramenta
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM tentou chamar {len(tool_calls)} ferramentas, executando apenas a primeira: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Construir dica de ferramentas não utilizadas
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list="、".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Situação 3: Sem chamada de ferramenta nem Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Número insuficiente de chamadas de ferramentas, recomendar ferramentas não utilizadas
                unused_tools = all_tools - used_tools
                unused_hint = f"(Estas ferramentas ainda não foram usadas, recomenda-se experimentá-las: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Chamadas de ferramentas suficientes, LLM gerou conteúdo mas sem prefixo "Final Answer:"
            # Adotar diretamente esta saída como resposta final, sem iterações desnecessárias
            logger.info(f"Seção {section.title} não detectou prefixo 'Final Answer:', adotando saída do LLM diretamente como conteúdo final (chamadas de ferramenta: {tool_calls_count} vezes)")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Atingiu número máximo de iterações, forçar geração de conteúdo
        logger.warning(f"Seção {section.title} atingiu número máximo de iterações, forçando geração")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Verificar se o LLM retornou None durante a finalização forçada
        if response is None:
            logger.error(f"Seção {section.title} LLM retornou None durante finalização forçada, usando mensagem de erro padrão")
            final_answer = f"(Falha na geração desta seção: LLM retornou resposta vazia, por favor tente novamente)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Registrar log de conclusão da geração do conteúdo da seção
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Gerar relatório completo (saída em tempo real por seção)
        
        Cada seção é salva em arquivo imediatamente após ser gerada, sem precisar aguardar o relatório inteiro.
        Estrutura de arquivos:
        reports/{report_id}/
            meta.json       - Metainformações do relatório
            outline.json    - Esboço do relatório
            progress.json   - Progresso da geração
            section_01.md   - Seção 1
            section_02.md   - Seção 2
            ...
            full_report.md  - Relatório completo
        
        Args:
            progress_callback: Função de callback de progresso (stage, progress, message)
            report_id: ID do relatório (opcional, gerado automaticamente se não fornecido)
            
        Returns:
            Report: Relatório completo
        """
        import uuid
        
        # Se não foi fornecido report_id, gerar automaticamente
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # Lista de títulos de seções concluídas (para acompanhamento de progresso)
        completed_section_titles = []
        
        try:
            # Inicialização: criar pasta do relatório e salvar estado inicial
            ReportManager._ensure_report_folder(report_id)
            
            # Inicializar registrador de logs (log estruturado agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Inicializar registrador de logs de console (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "Inicializando relatório...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Fase 1: Planejar esboço
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Iniciando planejamento do esboço do relatório...",
                completed_sections=[]
            )
            
            # Registrar log de início do planejamento
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Iniciando planejamento do esboço do relatório...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Registrar log de conclusão do planejamento
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Salvar esboço em arquivo
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Esboço concluído, {len(outline.sections)} seções",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"Esboço salvo em arquivo: {report_id}/outline.json")
            
            # Fase 2: Gerar seções sequencialmente (salvar por seção)
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Salvar conteúdo para contexto
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Atualizar progresso
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Gerando seção: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"Gerando seção: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Gerar conteúdo principal da seção
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Salvar seção
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Registrar log de conclusão da seção
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Seção salva: {report_id}/section_{section_num:02d}.md")
                
                # Atualizar progresso
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"Seção {section.title} concluída",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Fase 3: Montar relatório completo
            if progress_callback:
                progress_callback("generating", 95, "Montando relatório completo...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "Montando relatório completo...",
                completed_sections=completed_section_titles
            )
            
            # Usar ReportManager para montar relatório completo
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Calcular tempo total
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Registrar log de conclusão do relatório
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Salvar relatório final
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Geração do relatório concluída",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "Geração do relatório concluída")
            
            logger.info(f"Geração do relatório concluída: {report_id}")
            
            # Fechar registrador de logs de console
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"Falha na geração do relatório: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Registrar log de erro
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Salvar estado de falha
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Falha na geração do relatório: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Ignorar erros de salvamento
            
            # Fechar registrador de logs de console
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Conversar com o Report Agent
        
        Durante a conversa, o Agent pode chamar ferramentas de recuperação autonomamente para responder perguntas
        
        Args:
            message: Mensagem do usuário
            chat_history: Histórico de conversa
            
        Returns:
            {
                "response": "Resposta do Agent",
                "tool_calls": [lista de ferramentas chamadas],
                "sources": [fontes de informação]
            }
        """
        logger.info(f"Conversa com Report Agent: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # Obter conteúdo do relatório já gerado
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limitar tamanho do relatório para evitar contexto muito longo
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [conteúdo do relatório truncado] ..."
        except Exception as e:
            logger.warning(f"Falha ao obter conteúdo do relatório: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(sem relatório por enquanto)",
            tools_description=self._get_tools_description(),
        )

        # Construir mensagens
        messages = [{"role": "system", "content": system_prompt}]
        
        # Adicionar histórico de conversa
        for h in chat_history[-10:]:  # Limitar tamanho do histórico
            messages.append(h)
        
        # Adicionar mensagem do usuário
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # Ciclo ReACT (versão simplificada)
        tool_calls_made = []
        max_iterations = 2  # Reduzir número de rodadas
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Analisar chamadas de ferramentas
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # Sem chamadas de ferramentas, retornar resposta diretamente
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Executar chamadas de ferramentas (limitar quantidade)
            tool_results = []
            for call in tool_calls[:1]:  # No máximo 1 chamada de ferramenta por rodada
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Limitar tamanho do resultado
                })
                tool_calls_made.append(call)
            
            # Adicionar resultados às mensagens
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[Resultado de {r['tool']}]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Atingiu número máximo de iterações, obter resposta final
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Limpar resposta
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Gerenciador de Relatórios
    
    Responsável pelo armazenamento persistente e recuperação de relatórios
    
    Estrutura de arquivos (saída por seção):
    reports/
      {report_id}/
        meta.json          - Metainformações e status do relatório
        outline.json       - Esboço do relatório
        progress.json      - Progresso da geração
        section_01.md      - Seção 1
        section_02.md      - Seção 2
        ...
        full_report.md     - Relatório completo
    """
    
    # Diretório de armazenamento de relatórios
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Garantir que o diretório raiz de relatórios exista"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Obter caminho da pasta do relatório"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Garantir que a pasta do relatório exista e retornar o caminho"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Obter caminho do arquivo de metainformações do relatório"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Obter caminho do arquivo Markdown do relatório completo"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Obter caminho do arquivo de esboço"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Obter caminho do arquivo de progresso"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Obter caminho do arquivo Markdown da seção"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Obter caminho do arquivo de log do Agent"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Obter caminho do arquivo de log de console"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Obter conteúdo do log de console
        
        Este é o log de saída do console durante a geração do relatório (INFO, WARNING, etc.),
        diferente do log estruturado agent_log.jsonl.
        
        Args:
            report_id: ID do relatório
            from_line: A partir de qual linha começar a leitura (para obtenção incremental, 0 = desde o início)
            
        Returns:
            {
                "logs": [lista de linhas de log],
                "total_lines": número total de linhas,
                "from_line": linha inicial,
                "has_more": se há mais logs
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Manter linha de log original, removendo quebra de linha final
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Leitura até o final
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Obter log de console completo (obter tudo de uma vez)
        
        Args:
            report_id: ID do relatório
            
        Returns:
            Lista de linhas de log
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Obter conteúdo do log do Agent
        
        Args:
            report_id: ID do relatório
            from_line: A partir de qual linha começar a leitura (para obtenção incremental, 0 = desde o início)
            
        Returns:
            {
                "logs": [lista de entradas de log],
                "total_lines": número total de linhas,
                "from_line": linha inicial,
                "has_more": se há mais logs
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Pular linhas com falha na análise
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Leitura até o final
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obter log completo do Agent (obter tudo de uma vez)
        
        Args:
            report_id: ID do relatório
            
        Returns:
            Lista de entradas de log
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Salvar esboço do relatório
        
        Chamado imediatamente após a conclusão da fase de planejamento
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Esboço salvo: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Salvar seção individual

        Chamado imediatamente após a geração de cada seção, implementando saída por seção

        Args:
            report_id: ID do relatório
            section_index: Índice da seção (começando em 1)
            section: Objeto da seção

        Returns:
            Caminho do arquivo salvo
        """
        cls._ensure_report_folder(report_id)

        # Construir conteúdo Markdown da seção - limpar possíveis títulos duplicados
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Salvar arquivo
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Seção salva: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Limpar conteúdo da seção
        
        1. Remover linhas de título Markdown duplicadas no início do conteúdo
        2. Converter todos os títulos ### e abaixo em texto em negrito
        
        Args:
            content: Conteúdo original
            section_title: Título da seção
            
        Returns:
            Conteúdo limpo
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Verificar se é linha de título Markdown
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Verificar se é título duplicado com o título da seção (pular duplicatas nas primeiras 5 linhas)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Converter todos os níveis de título (#, ##, ###, #### etc.) em negrito
                # Pois o título da seção é adicionado pelo sistema, o conteúdo não deve ter nenhum título
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Adicionar linha em branco
                continue
            
            # Se a linha anterior foi um título pulado e a linha atual é vazia, pular também
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Remover linhas em branco do início
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Remover linhas separadoras do início
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Remover também linhas em branco após a linha separadora
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Atualizar progresso da geração do relatório
        
        O frontend pode obter o progresso em tempo real lendo progress.json
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
                """Obter progresso da geração do relatório"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obter lista de seções já geradas
        
        Retorna informações de todos os arquivos de seção salvos
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extrair índice da seção a partir do nome do arquivo
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Montar relatório completo
        
        Montar relatório completo a partir dos arquivos de seção salvos, com limpeza de títulos
        """
        folder = cls._get_report_folder(report_id)
        
        # Construir cabeçalho do relatório
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Ler todos os arquivos de seção em ordem
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Pós-processamento: limpar problemas de títulos em todo o relatório
        md_content = cls._post_process_report(md_content, outline)
        
        # Salvar relatório completo
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Relatório completo montado: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Pós-processar conteúdo do relatório
        
        1. Remover títulos duplicados
        2. Manter título principal do relatório (#) e títulos de seção (##), remover outros níveis (###, #### etc.)
        3. Limpar linhas em branco e separadores excedentes
        
        Args:
            content: Conteúdo original do relatório
            outline: Esboço do relatório
            
        Returns:
            Conteúdo processado
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Coletar todos os títulos de seção do esboço
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Verificar se é linha de título
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Verificar se é título duplicado (mesmo conteúdo nas últimas 5 linhas)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Pular título duplicado e linhas em branco posteriores
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Tratamento de nível de título:
                # - # (level=1) manter apenas o título principal do relatório
                # - ## (level=2) manter títulos de seção
                # - ### e abaixo (level>=3) converter em texto em negrito
                
                if level == 1:
                    if title == outline.title:
                        # Manter título principal do relatório
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Título de seção usando # incorretamente, corrigir para ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Outros títulos de nível 1 converter em negrito
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Manter título de seção
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Título de nível 2 que não é seção, converter em negrito
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # Títulos ### e abaixo converter em texto em negrito
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Pular linha separadora logo após um título
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Manter apenas uma linha em branco após o título
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Limpar múltiplas linhas em branco consecutivas (manter no máximo 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
                """Salvar metainformações e relatório completo"""
        cls._ensure_report_folder(report.report_id)
        
        # Salvar metainformações em JSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Salvar esboço
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Salvar relatório Markdown completo
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"Relatório salvo: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
                """Obter relatório"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Compatibilidade com formato antigo: verificar arquivos armazenados diretamente no diretório reports
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruir objeto Report
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # Se markdown_content estiver vazio, tentar ler de full_report.md
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
                """Obter relatório por ID de simulação"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Formato novo: pasta
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Compatibilidade com formato antigo: arquivo JSON
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
                """Listar relatórios"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Formato novo: pasta
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Compatibilidade com formato antigo: arquivo JSON
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Ordenar por data de criação em ordem decrescente
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
                """Excluir relatório (pasta inteira)"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # Formato novo: excluir pasta inteira
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Pasta do relatório excluída: {report_id}")
            return True
        
        # Compatibilidade com formato antigo: excluir arquivos individuais
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
