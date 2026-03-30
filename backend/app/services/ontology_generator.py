"""
Serviço de geração de ontologia
Interface 1: Analisar conteúdo de texto e gerar definições de tipos de entidades e relações adequadas para simulação social
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# Prompt de sistema para geração de ontologia
ONTOLOGY_SYSTEM_PROMPT = """Você é um especialista profissional em design de ontologias de grafos de conhecimento. Sua tarefa é analisar o conteúdo textual fornecido e os requisitos de simulação, projetando tipos de entidades e tipos de relações adequados para **simulação de opinião pública em mídias sociais**.

**Importante: você deve gerar dados em formato JSON válido, sem incluir qualquer outro conteúdo.**

## Contexto da Tarefa Principal

Estamos construindo um **sistema de simulação de opinião pública em mídias sociais**. Neste sistema:
- Cada entidade é uma "conta" ou "ator" que pode se manifestar, interagir e disseminar informações nas mídias sociais
- As entidades influenciam umas às outras, compartilham, comentam e respondem
- Precisamos simular as reações das partes envolvidas em eventos de opinião pública e os caminhos de disseminação de informação

Portanto, **as entidades devem ser atores reais que existem na realidade e que podem se manifestar e interagir nas mídias sociais**:

**Podem ser**:
- Indivíduos específicos (figuras públicas, partes envolvidas, formadores de opinião, especialistas, pessoas comuns)
- Empresas (incluindo suas contas oficiais)
- Organizações e instituições (universidades, associações, ONGs, sindicatos, etc.)
- Departamentos governamentais, órgãos reguladores
- Veículos de mídia (jornais, emissoras de TV, mídia independente, sites)
- As próprias plataformas de mídias sociais
- Representantes de grupos específicos (como associações de ex-alunos, fã-clubes, grupos de defesa de direitos, etc.)

**Não podem ser**:
- Conceitos abstratos (como "opinião pública", "emoção", "tendência")
- Temas/tópicos (como "integridade acadêmica", "reforma educacional")
- Opiniões/posições (como "lado a favor", "lado contra")

## Formato de Saída

Por favor, gere em formato JSON com a seguinte estrutura:

```json
{
    "entity_types": [
        {
            "name": "Nome do tipo de entidade (inglês, PascalCase)",
            "description": "Descrição breve (inglês, máximo 100 caracteres)",
            "attributes": [
                {
                    "name": "Nome do atributo (inglês, snake_case)",
                    "type": "text",
                    "description": "Descrição do atributo"
                }
            ],
            "examples": ["Entidade exemplo 1", "Entidade exemplo 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Nome do tipo de relação (inglês, UPPER_SNAKE_CASE)",
            "description": "Descrição breve (inglês, máximo 100 caracteres)",
            "source_targets": [
                {"source": "Tipo de entidade de origem", "target": "Tipo de entidade de destino"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Breve análise do conteúdo do texto (em português)"
}
```

## Diretrizes de Design (Extremamente Importante!)

### 1. Design de Tipos de Entidade - Deve ser rigorosamente seguido

**Requisito de quantidade: deve haver exatamente 10 tipos de entidade**

**Requisito de estrutura hierárquica (deve incluir tipos específicos e tipos genéricos de fallback)**:

Seus 10 tipos de entidade devem incluir os seguintes níveis:

A. **Tipos genéricos de fallback (obrigatórios, devem ser os 2 últimos da lista)**:
   - `Person`: Tipo genérico para qualquer pessoa física. Quando uma pessoa não se encaixa em outros tipos mais específicos, é classificada aqui.
   - `Organization`: Tipo genérico para qualquer organização. Quando uma organização não se encaixa em outros tipos mais específicos, é classificada aqui.

B. **Tipos específicos (8, projetados com base no conteúdo do texto)**:
   - Projete tipos mais específicos para os papéis principais que aparecem no texto
   - Exemplo: se o texto envolve eventos acadêmicos, pode ter `Student`, `Professor`, `University`
   - Exemplo: se o texto envolve eventos comerciais, pode ter `Company`, `CEO`, `Employee`

**Por que são necessários tipos genéricos de fallback**:
- Diversos personagens aparecerão no texto, como "professor do ensino fundamental", "transeunte", "algum internauta"
- Se não houver um tipo específico correspondente, eles devem ser classificados como `Person`
- Da mesma forma, pequenas organizações, grupos temporários, etc. devem ser classificados como `Organization`

**Princípios de design de tipos específicos**:
- Identifique os tipos de papéis que aparecem com alta frequência ou são fundamentais no texto
- Cada tipo específico deve ter limites claros, evitando sobreposição
- A description deve explicar claramente a diferença entre este tipo e o tipo genérico de fallback

### 2. Design de Tipos de Relação

- Quantidade: 6-10
- As relações devem refletir conexões reais nas interações de mídias sociais
- Garanta que os source_targets das relações cubram os tipos de entidade que você definiu

### 3. Design de Atributos

- 1-3 atributos-chave por tipo de entidade
- **Atenção**: nomes de atributos não podem usar `name`, `uuid`, `group_id`, `created_at`, `summary` (são palavras reservadas do sistema)
- Recomendado usar: `full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Referência de Tipos de Entidade

**Categoria Pessoa (específicos)**:
- Student: Estudante
- Professor: Professor/Acadêmico
- Journalist: Jornalista
- Celebrity: Celebridade/Influenciador
- Executive: Executivo
- Official: Funcionário do governo
- Lawyer: Advogado
- Doctor: Médico

**Categoria Pessoa (genérico de fallback)**:
- Person: Qualquer pessoa física (usado quando não se encaixa nos tipos específicos acima)

**Categoria Organização (específicos)**:
- University: Universidade
- Company: Empresa
- GovernmentAgency: Órgão governamental
- MediaOutlet: Veículo de mídia
- Hospital: Hospital
- School: Escola de ensino fundamental/médio
- NGO: Organização não governamental

**Categoria Organização (genérico de fallback)**:
- Organization: Qualquer organização (usado quando não se encaixa nos tipos específicos acima)

## Referência de Tipos de Relação

- WORKS_FOR: Trabalha em
- STUDIES_AT: Estuda em
- AFFILIATED_WITH: Afiliado a
- REPRESENTS: Representa
- REGULATES: Regulamenta
- REPORTS_ON: Reporta sobre
- COMMENTS_ON: Comenta sobre
- RESPONDS_TO: Responde a
- SUPPORTS: Apoia
- OPPOSES: Opõe-se a
- COLLABORATES_WITH: Colabora com
- COMPETES_WITH: Compete com
"""


class OntologyGenerator:
    """
    Gerador de ontologia
    Analisa conteúdo de texto e gera definições de tipos de entidades e relações
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Gerar definição de ontologia
        
        Args:
            document_texts: Lista de textos dos documentos
            simulation_requirement: Descrição dos requisitos de simulação
            additional_context: Contexto adicional
            
        Returns:
            Definição de ontologia (entity_types, edge_types, etc.)
        """
        # Construir mensagem do usuário
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # Chamar LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # Validar e pós-processar
        result = self._validate_and_process(result)
        
        return result
    
    # Comprimento máximo do texto enviado ao LLM (50 mil caracteres)
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Construir mensagem do usuário"""
        
        # Combinar textos
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # Se o texto exceder 50 mil caracteres, truncar (afeta apenas o conteúdo enviado ao LLM, não a construção do grafo)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Texto original com {original_length} caracteres, truncado para os primeiros {self.MAX_TEXT_LENGTH_FOR_LLM} caracteres para análise de ontologia)..."
        
        message = f"""## Requisitos de Simulação

{simulation_requirement}

## Conteúdo do Documento

{combined_text}
"""
        
        if additional_context:
            message += f"""
## Observações Adicionais

{additional_context}
"""
        
        message += """
Com base no conteúdo acima, projete tipos de entidades e tipos de relações adequados para simulação de opinião pública.

**Regras obrigatórias**:
1. Deve gerar exatamente 10 tipos de entidade
2. Os 2 últimos devem ser tipos genéricos de fallback: Person (fallback para pessoa) e Organization (fallback para organização)
3. Os 8 primeiros são tipos específicos projetados com base no conteúdo do texto
4. Todos os tipos de entidade devem ser atores reais que podem se manifestar, não conceitos abstratos
5. Nomes de atributos não podem usar palavras reservadas como name, uuid, group_id, etc. Use full_name, org_name, etc.
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validar e pós-processar resultado"""
        
        # Garantir que os campos necessários existam
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # Validar tipos de entidade
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Garantir que a description não exceda 100 caracteres
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # Validar tipos de relação
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Limite da API Zep: máximo de 10 tipos de entidade customizados, máximo de 10 tipos de aresta customizados
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        
        # Definição dos tipos genéricos de fallback
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # Verificar se já existem tipos genéricos de fallback
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # Tipos genéricos de fallback que precisam ser adicionados
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # Se adicionar exceder 10, é necessário remover alguns tipos existentes
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calcular quantos precisam ser removidos
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Remover do final (preservar os tipos específicos mais importantes no início)
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # Adicionar tipos genéricos de fallback
            result["entity_types"].extend(fallbacks_to_add)
        
        # Garantir que não exceda o limite final (programação defensiva)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Converter definição de ontologia em código Python (similar a ontology.py)
        
        Args:
            ontology: Definição de ontologia
            
        Returns:
            String de código Python
        """
        code_lines = [
            '"""',
            'Definição de tipos de entidade customizados',
            'Gerado automaticamente pelo MiroFish para simulação de opinião pública',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Definição de Tipos de Entidade ==============',
            '',
        ]
        
        # Gerar tipos de entidade
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== Definição de Tipos de Relação ==============')
        code_lines.append('')
        
        # Gerar tipos de relação
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Converter para nome de classe PascalCase
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # Gerar dicionário de tipos
        code_lines.append('# ============== Configuração de Tipos ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # Gerar mapeamento de source_targets das arestas
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

