"""
Gerador de Agent Profile OASIS
Converte entidades do grafo Zep para o formato de Agent Profile necessário pela plataforma de simulação OASIS

Melhorias de otimização:
1. Chama a funcionalidade de busca do Zep para enriquecer informações dos nós
2. Otimiza prompts para gerar personas muito detalhadas
3. Diferencia entidades individuais de entidades de grupo abstratas
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.oasis_profile')


@dataclass
class OasisAgentProfile:
    """Estrutura de dados do Agent Profile OASIS"""
    # Campos gerais
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str
    
    # Campos opcionais - estilo Reddit
    karma: int = 1000
    
    # Campos opcionais - estilo Twitter
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500
    
    # Informações adicionais de persona
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)
    
    # Informações da entidade de origem
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None
    
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    def to_reddit_format(self) -> Dict[str, Any]:
        """Converter para formato da plataforma Reddit"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # A biblioteca OASIS exige o nome do campo como username (sem underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }
        
        # Adicionar informações extras de persona (se houver)
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_twitter_format(self) -> Dict[str, Any]:
        """Converter para formato da plataforma Twitter"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # A biblioteca OASIS exige o nome do campo como username (sem underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }
        
        # Adicionar informações extras de persona
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Converter para formato de dicionário completo"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    Gerador de Profile OASIS
    
    Converte entidades do grafo Zep para Agent Profiles necessários pela simulação OASIS
    
    Características otimizadas:
    1. Chama funcionalidade de busca no grafo Zep para obter contexto mais rico
    2. Gera personas muito detalhadas (incluindo informações básicas, experiência profissional, traços de personalidade, comportamento em redes sociais, etc.)
    3. Diferencia entidades individuais de entidades de grupo abstratas
    """
    
    # Lista de tipos MBTI
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    
    # Lista de países comuns
    COUNTRIES = [
        "China", "US", "UK", "Japan", "Germany", "France", 
        "Canada", "Australia", "Brazil", "India", "South Korea"
    ]
    
    # Entidades do tipo individual (necessitam gerar persona específica)
    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure", 
        "expert", "faculty", "official", "journalist", "activist"
    ]
    
    # Entidades do tipo grupo/instituição (necessitam gerar persona representativa do grupo)
    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo", 
        "mediaoutlet", "company", "institution", "group", "community"
    ]
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        zep_api_key: Optional[str] = None,
        graph_id: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY não configurada")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Cliente Zep usado para buscar contexto enriquecido
        self.zep_api_key = zep_api_key or Config.ZEP_API_KEY
        self.zep_client = None
        self.graph_id = graph_id
        
        if self.zep_api_key:
            try:
                self.zep_client = Zep(api_key=self.zep_api_key)
            except Exception as e:
                logger.warning(f"Falha ao inicializar cliente Zep: {e}")
    
    def generate_profile_from_entity(
        self, 
        entity: EntityNode, 
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        Gerar OASIS Agent Profile a partir de entidade Zep
        
        Args:
            entity: Nó de entidade Zep
            user_id: ID do usuário (para OASIS)
            use_llm: Se deve usar LLM para gerar persona detalhada
            
        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"
        
        # Informações básicas
        name = entity.name
        user_name = self._generate_username(name)
        
        # Construir informações de contexto
        context = self._build_entity_context(entity)
        
        if use_llm:
            # Usar LLM para gerar persona detalhada
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:
            # Usar regras para gerar persona básica
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )
        
        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )
    
    def _generate_username(self, name: str) -> str:
        """Gerar nome de usuário"""
        # Remover caracteres especiais, converter para minúsculas
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')
        
        # Adicionar sufixo aleatório para evitar duplicatas
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"
    
    def _search_zep_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Usar funcionalidade de busca híbrida do grafo Zep para obter informações ricas relacionadas à entidade
        
        O Zep não tem interface de busca híbrida integrada, é necessário buscar edges e nodes separadamente e depois mesclar os resultados.
        Usa requisições paralelas para buscar simultaneamente, melhorando a eficiência.
        
        Args:
            entity: Objeto do nó de entidade
            
        Returns:
            Dicionário contendo facts, node_summaries, context
        """
        import concurrent.futures
        
        if not self.zep_client:
            return {"facts": [], "node_summaries": [], "context": ""}
        
        entity_name = entity.name
        
        results = {
            "facts": [],
            "node_summaries": [],
            "context": ""
        }
        
        # É necessário ter graph_id para realizar a busca
        if not self.graph_id:
            logger.debug(f"Busca Zep ignorada: graph_id não definido")
            return results
        
        comprehensive_query = f"Todas as informações, atividades, eventos, relações e contexto sobre {entity_name}"
        
        def search_edges():
            """Buscar arestas (fatos/relações) - com mecanismo de retry"""
            max_retries = 3
            last_exception = None
            delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=30,
                        scope="edges",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Busca de arestas Zep falhou na tentativa {attempt + 1}: {str(e)[:80]}, tentando novamente...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Busca de arestas Zep falhou após {max_retries} tentativas: {e}")
            return None
        
        def search_nodes():
            """Buscar nós (resumos de entidades) - com mecanismo de retry"""
            max_retries = 3
            last_exception = None
            delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=20,
                        scope="nodes",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Busca de nós Zep falhou na tentativa {attempt + 1}: {str(e)[:80]}, tentando novamente...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Busca de nós Zep falhou após {max_retries} tentativas: {e}")
            return None
        
        try:
            # Executar busca de edges e nodes em paralelo
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)
                
                # Obter resultados
                edge_result = edge_future.result(timeout=30)
                node_result = node_future.result(timeout=30)
            
            # Processar resultados da busca de arestas
            all_facts = set()
            if edge_result and hasattr(edge_result, 'edges') and edge_result.edges:
                for edge in edge_result.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        all_facts.add(edge.fact)
            results["facts"] = list(all_facts)
            
            # Processar resultados da busca de nós
            all_summaries = set()
            if node_result and hasattr(node_result, 'nodes') and node_result.nodes:
                for node in node_result.nodes:
                    if hasattr(node, 'summary') and node.summary:
                        all_summaries.add(node.summary)
                    if hasattr(node, 'name') and node.name and node.name != entity_name:
                        all_summaries.add(f"Entidade relacionada: {node.name}")
            results["node_summaries"] = list(all_summaries)
            
            # Construir contexto abrangente
            context_parts = []
            if results["facts"]:
                context_parts.append("Informações factuais:\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                context_parts.append("Entidades relacionadas:\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)
            
            logger.info(f"Busca híbrida Zep concluída: {entity_name}, obtidos {len(results['facts'])} fatos, {len(results['node_summaries'])} nós relacionados")
            
        except concurrent.futures.TimeoutError:
            logger.warning(f"Busca Zep expirou ({entity_name})")
        except Exception as e:
            logger.warning(f"Busca Zep falhou ({entity_name}): {e}")
        
        return results
    
    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Construir informações de contexto completas da entidade
        
        Inclui:
        1. Informações de arestas da própria entidade (fatos)
        2. Informações detalhadas dos nós relacionados
        3. Informações enriquecidas obtidas pela busca híbrida do Zep
        """
        context_parts = []
        
        # 1. Adicionar informações de atributos da entidade
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Atributos da entidade\n" + "\n".join(attrs))
        
        # 2. Adicionar informações de arestas relacionadas (fatos/relações)
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:  # Sem limite de quantidade
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")
                
                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (entidade relacionada)")
                    else:
                        relationships.append(f"- (entidade relacionada) --[{edge_name}]--> {entity.name}")
            
            if relationships:
                context_parts.append("### Fatos e relações relacionados\n" + "\n".join(relationships))
        
        # 3. Adicionar informações detalhadas dos nós relacionados
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:  # Sem limite de quantidade
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")
                
                # Filtrar labels padrão
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""
                
                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")
            
            if related_info:
                context_parts.append("### Informações de entidades relacionadas\n" + "\n".join(related_info))
        
        # 4. Usar busca híbrida do Zep para obter informações mais ricas
        zep_results = self._search_zep_for_entity(entity)
        
        if zep_results.get("facts"):
            # Deduplicar: excluir fatos já existentes
            new_facts = [f for f in zep_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### Informações factuais encontradas pelo Zep\n" + "\n".join(f"- {f}" for f in new_facts[:15]))
        
        if zep_results.get("node_summaries"):
            context_parts.append("### Nós relacionados encontrados pelo Zep\n" + "\n".join(f"- {s}" for s in zep_results["node_summaries"][:10]))
        
        return "\n\n".join(context_parts)
    
    def _is_individual_entity(self, entity_type: str) -> bool:
        """Verificar se é uma entidade do tipo individual"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES
    
    def _is_group_entity(self, entity_type: str) -> bool:
        """Verificar se é uma entidade do tipo grupo/instituição"""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES
    
    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """
        Usar LLM para gerar persona muito detalhada
        
        Distingue por tipo de entidade:
        - Entidade individual: gera configuração de personagem específica
        - Entidade de grupo/instituição: gera configuração de conta representativa
        """
        
        is_individual = self._is_individual_entity(entity_type)
        
        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        # Tentar gerar múltiplas vezes, até sucesso ou atingir o número máximo de tentativas
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # Reduzir temperatura a cada retry
                    # Não definir max_tokens, deixar o LLM gerar livremente
                )
                
                content = response.choices[0].message.content
                
                # Verificar se foi truncado (finish_reason não é 'stop')
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"Saída do LLM truncada (attempt {attempt+1}), tentando corrigir...")
                    content = self._fix_truncated_json(content)
                
                # Tentar parsear JSON
                try:
                    result = json.loads(content)
                    
                    # Validar campos obrigatórios
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name} é um(a) {entity_type}."
                    
                    return result
                    
                except json.JSONDecodeError as je:
                    logger.warning(f"Falha no parse do JSON (attempt {attempt+1}): {str(je)[:80]}")
                    
                    # Tentar corrigir o JSON
                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result
                    
                    last_error = je
                    
            except Exception as e:
                logger.warning(f"Chamada LLM falhou (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))  # Backoff exponencial
        
        logger.warning(f"Falha ao gerar persona via LLM ({max_attempts} tentativas): {last_error}, usando geração por regras")
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )
    
    def _fix_truncated_json(self, content: str) -> str:
        """Corrigir JSON truncado (saída truncada pelo limite de max_tokens)"""
        import re
        
        # Se o JSON foi truncado, tentar fechá-lo
        content = content.strip()
        
        # Calcular parênteses não fechados
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Verificar se há strings não fechadas
        # Verificação simples: se após a última aspas não há vírgula ou parêntese de fechamento, a string pode ter sido truncada
        if content and content[-1] not in '",}]':
            # Tentar fechar a string
            content += '"'
        
        # Fechar parênteses
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        """Tentar corrigir JSON corrompido"""
        import re
        
        # 1. Primeiro tentar corrigir caso truncado
        content = self._fix_truncated_json(content)
        
        # 2. Tentar extrair parte JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # 3. Tratar problema de quebras de linha em strings
            # Encontrar todos os valores de string e substituir quebras de linha
            def fix_string_newlines(match):
                s = match.group(0)
                # Substituir quebras de linha reais na string por espaços
                s = s.replace('\n', ' ').replace('\r', ' ')
                # Substituir espaços extras
                s = re.sub(r'\s+', ' ', s)
                return s
            
            # Corresponder valores de string JSON
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)
            
            # 4. Tentar parsear
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. Se ainda falhar, tentar correção mais agressiva
                try:
                    # Remover todos os caracteres de controle
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                    # Substituir todos os espaços em branco consecutivos
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass
        
        # 6. Tentar extrair informações parciais do conteúdo
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # Pode ter sido truncado
        
        bio = bio_match.group(1) if bio_match else (entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}")
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name} é um(a) {entity_type}.")
        
        # Se conteúdo significativo foi extraído, marcar como corrigido
        if bio_match or persona_match:
            logger.info(f"Informações parciais extraídas do JSON corrompido")
            return {
                "bio": bio,
                "persona": persona,
                "_fixed": True
            }
        
        # 7. Falha total, retornar estrutura básica
        logger.warning(f"Correção do JSON falhou, retornando estrutura básica")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} é um(a) {entity_type}."
        }
    
    def _get_system_prompt(self, is_individual: bool) -> str:
        """Obter prompt do sistema"""
        base_prompt = "Você é um especialista em geração de perfis de usuários de mídias sociais. Gere personas detalhadas e realistas para simulação de opinião pública, reproduzindo ao máximo a situação real existente. Deve retornar em formato JSON válido, todos os valores de string não podem conter quebras de linha não escapadas. Use português brasileiro."
        return base_prompt
    
    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Construir prompt detalhado de persona para entidade individual"""
        
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "Nenhum"
        context_str = context[:3000] if context else "Sem contexto adicional"
        
        return f"""Gere uma persona detalhada de usuário de mídia social para a entidade, reproduzindo ao máximo a situação real existente.

Nome da entidade: {entity_name}
Tipo da entidade: {entity_type}
Resumo da entidade: {entity_summary}
Atributos da entidade: {attrs_str}

Informações de contexto:
{context_str}

Gere um JSON contendo os seguintes campos:

1. bio: Biografia de mídia social, 200 caracteres
2. persona: Descrição detalhada da persona (2000 caracteres em texto puro), deve incluir:
   - Informações básicas (idade, profissão, formação acadêmica, localização)
   - Histórico do personagem (experiências importantes, relação com eventos, relações sociais)
   - Traços de personalidade (tipo MBTI, personalidade central, forma de expressão emocional)
   - Comportamento em mídias sociais (frequência de postagens, preferências de conteúdo, estilo de interação, características linguísticas)
   - Posições e opiniões (atitude sobre temas, conteúdo que pode irritar/emocionar)
   - Características únicas (bordões, experiências especiais, hobbies pessoais)
   - Memória pessoal (parte importante da persona, deve apresentar a relação deste indivíduo com os eventos, e as ações e reações já existentes deste indivíduo nos eventos)
3. age: Número da idade (deve ser inteiro)
4. gender: Gênero, deve ser em inglês: "male" ou "female"
5. mbti: Tipo MBTI (como INTJ, ENFP, etc.)
6. country: País (use português, como "Brasil")
7. profession: Profissão
8. interested_topics: Array de tópicos de interesse

Importante:
- Todos os valores de campos devem ser strings ou números, não use quebras de linha
- persona deve ser uma descrição textual coerente e contínua
- Use português brasileiro (exceto o campo gender que deve ser em inglês male/female)
- O conteúdo deve ser consistente com as informações da entidade
- age deve ser um inteiro válido, gender deve ser "male" ou "female"
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Construir prompt detalhado de persona para entidade de grupo/instituição"""
        
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "Nenhum"
        context_str = context[:3000] if context else "Sem contexto adicional"
        
        return f"""Gere uma configuração detalhada de conta de mídia social para a entidade institucional/grupal, reproduzindo ao máximo a situação real existente.

Nome da entidade: {entity_name}
Tipo da entidade: {entity_type}
Resumo da entidade: {entity_summary}
Atributos da entidade: {attrs_str}

Informações de contexto:
{context_str}

Gere um JSON contendo os seguintes campos:

1. bio: Biografia da conta oficial, 200 caracteres, profissional e adequada
2. persona: Descrição detalhada da configuração da conta (2000 caracteres em texto puro), deve incluir:
   - Informações básicas da instituição (nome oficial, natureza institucional, contexto de fundação, funções principais)
   - Posicionamento da conta (tipo de conta, público-alvo, função central)
   - Estilo de comunicação (características linguísticas, expressões comuns, tópicos proibidos)
   - Características de publicação (tipos de conteúdo, frequência de publicação, horários de atividade)
   - Posição e atitude (posição oficial sobre tópicos centrais, forma de lidar com controvérsias)
   - Notas especiais (perfil do grupo representado, hábitos de operação)
   - Memória institucional (parte importante da persona institucional, deve apresentar a relação desta instituição com os eventos, e as ações e reações já existentes desta instituição nos eventos)
3. age: Fixo em 30 (idade virtual da conta institucional)
4. gender: Fixo em "other" (conta institucional usa other para indicar não-pessoal)
5. mbti: Tipo MBTI, usado para descrever o estilo da conta, como ISTJ representando rigoroso e conservador
6. country: País (use português, como "Brasil")
7. profession: Descrição da função institucional
8. interested_topics: Array de áreas de interesse

Importante:
- Todos os valores de campos devem ser strings ou números, não permitido valor null
- persona deve ser uma descrição textual coerente e contínua, não use quebras de linha
- Use português brasileiro (exceto o campo gender que deve ser em inglês "other")
- age deve ser inteiro 30, gender deve ser string "other"
- A comunicação da conta institucional deve estar de acordo com seu posicionamento"""
    
    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gerar persona básica usando regras"""
        
        # Gerar personas diferentes conforme o tipo de entidade
        entity_type_lower = entity_type.lower()
        
        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and connecting with peers.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }
        
        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }
        
        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,  # Idade virtual da instituição
                "gender": "other",  # Instituição usa other
                "mbti": "ISTJ",  # Estilo institucional: rigoroso e conservador
                "country": "中国",
                "profession": "Media",
                "interested_topics": ["General News", "Current Events", "Public Affairs"],
            }
        
        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"Official account of {entity_name}.",
                "persona": f"{entity_name} is an institutional entity that communicates official positions, announcements, and engages with stakeholders on relevant matters.",
                "age": 30,  # Idade virtual da instituição
                "gender": "other",  # Instituição usa other
                "mbti": "ISTJ",  # Estilo institucional: rigoroso e conservador
                "country": "中国",
                "profession": entity_type,
                "interested_topics": ["Public Policy", "Community", "Official Announcements"],
            }
        
        else:
            # Persona padrão
            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }
    
    def set_graph_id(self, graph_id: str):
        """Definir ID do grafo para busca no Zep"""
        self.graph_id = graph_id
    
    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit"
    ) -> List[OasisAgentProfile]:
        """
        Gerar Agent Profiles em lote a partir de entidades (com suporte a geração paralela)
        
        Args:
            entities: Lista de entidades
            use_llm: Se deve usar LLM para gerar persona detalhada
            progress_callback: Função de callback de progresso (current, total, message)
            graph_id: ID do grafo, usado para busca no Zep para obter contexto mais rico
            parallel_count: Quantidade de gerações paralelas, padrão 5
            realtime_output_path: Caminho do arquivo para escrita em tempo real (se fornecido, escreve a cada geração)
            output_platform: Formato da plataforma de saída ("reddit" ou "twitter")
            
        Returns:
            Lista de Agent Profiles
        """
        import concurrent.futures
        from threading import Lock
        
        # Definir graph_id para busca no Zep
        if graph_id:
            self.graph_id = graph_id
        
        total = len(entities)
        profiles = [None] * total  # Pré-alocar lista para manter ordem
        completed_count = [0]  # Usar lista para permitir modificação no closure
        lock = Lock()
        
        # Função auxiliar para escrita em tempo real no arquivo
        def save_profiles_realtime():
            """Salvar profiles já gerados no arquivo em tempo real"""
            if not realtime_output_path:
                return
            
            with lock:
                # Filtrar profiles já gerados
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return
                
                try:
                    if output_platform == "reddit":
                        # Formato JSON Reddit
                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Formato CSV Twitter
                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Falha ao salvar profiles em tempo real: {e}")
        
        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Função de trabalho para gerar um único profile"""
            entity_type = entity.get_entity_type() or "Entity"
            
            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )
                
                # Saída em tempo real da persona gerada para console e log
                self._print_generated_profile(entity.name, entity_type, profile)
                
                return idx, profile, None
                
            except Exception as e:
                logger.error(f"Falha ao gerar persona da entidade {entity.name}: {str(e)}")
                # Criar um profile básico
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)
        
        logger.info(f"Iniciando geração paralela de {total} personas de Agent (paralelismo: {parallel_count})...")
        print(f"\n{'='*60}")
        print(f"Iniciando geração de personas de Agent - {total} entidades, paralelismo: {parallel_count}")
        print(f"{'='*60}\n")
        
        # Executar em paralelo usando thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # Submeter todas as tarefas
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }
            
            # Coletar resultados
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"
                
                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile
                    
                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]
                    
                    # Escrever no arquivo em tempo real
                    save_profiles_realtime()
                    
                    if progress_callback:
                        progress_callback(
                            current, 
                            total, 
                            f"Concluído {current}/{total}: {entity.name} ({entity_type})"
                        )
                    
                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} usando persona de backup: {error}")
                    else:
                        logger.info(f"[{current}/{total}] Persona gerada com sucesso: {entity.name} ({entity_type})")
                        
                except Exception as e:
                    logger.error(f"Exceção ao processar entidade {entity.name}: {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    # Escrever no arquivo em tempo real (mesmo para persona de backup)
                    save_profiles_realtime()
        
        print(f"\n{'='*60}")
        print(f"Geração de personas concluída! Total gerado: {len([p for p in profiles if p])} Agents")
        print(f"{'='*60}\n")
        
        return profiles
    
    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        """Saída em tempo real da persona gerada para o console (conteúdo completo, sem truncamento)"""
        separator = "-" * 70
        
        # Construir conteúdo de saída completo (sem truncamento)
        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'Nenhum'
        
        output_lines = [
            f"\n{separator}",
            f"[Gerado] {entity_name} ({entity_type})",
            f"{separator}",
            f"Nome de usuário: {profile.user_name}",
            f"",
            f"【Biografia】",
            f"{profile.bio}",
            f"",
            f"【Persona Detalhada】",
            f"{profile.persona}",
            f"",
            f"【Atributos Básicos】",
            f"Idade: {profile.age} | Gênero: {profile.gender} | MBTI: {profile.mbti}",
            f"Profissão: {profile.profession} | País: {profile.country}",
            f"Tópicos de interesse: {topics_str}",
            separator
        ]
        
        output = "\n".join(output_lines)
        
        # Apenas saída no console (evitar duplicação, logger não exibe conteúdo completo)
        print(output)
    
    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """
        Salvar Profile em arquivo (escolhe o formato correto conforme a plataforma)
        
        Requisitos de formato da plataforma OASIS:
        - Twitter: formato CSV
        - Reddit: formato JSON
        
        Args:
            profiles: Lista de Profiles
            file_path: Caminho do arquivo
            platform: Tipo de plataforma ("reddit" ou "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)
    
    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Salvar Twitter Profile em formato CSV (conforme requisitos oficiais do OASIS)
        
        Campos CSV exigidos pelo OASIS Twitter:
        - user_id: ID do usuário (começando do 0 conforme ordem do CSV)
        - name: Nome real do usuário
        - username: Nome de usuário no sistema
        - user_char: Descrição detalhada da persona (injetada no prompt de sistema do LLM, guia o comportamento do Agent)
        - description: Biografia pública curta (exibida na página de perfil do usuário)
        
        Diferença entre user_char e description:
        - user_char: Uso interno, prompt de sistema do LLM, determina como o Agent pensa e age
        - description: Exibição externa, biografia visível para outros usuários
        """
        import csv
        
        # Garantir que a extensão do arquivo é .csv
        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Escrever cabeçalho exigido pelo OASIS
            headers = ['user_id', 'name', 'username', 'user_char', 'description']
            writer.writerow(headers)
            
            # Escrever linhas de dados
            for idx, profile in enumerate(profiles):
                # user_char: Persona completa (bio + persona), para prompt de sistema do LLM
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Tratar quebras de linha (substituir por espaços no CSV)
                user_char = user_char.replace('\n', ' ').replace('\r', ' ')
                
                # description: Biografia curta, para exibição externa
                description = profile.bio.replace('\n', ' ').replace('\r', ' ')
                
                row = [
                    idx,                    # user_id: ID sequencial começando do 0
                    profile.name,           # name: Nome real
                    profile.user_name,      # username: Nome de usuário
                    user_char,              # user_char: Persona completa (uso interno do LLM)
                    description             # description: Biografia curta (exibição externa)
                ]
                writer.writerow(row)
        
        logger.info(f"Salvos {len(profiles)} Twitter Profiles em {file_path} (formato CSV OASIS)")
    
    def _normalize_gender(self, gender: Optional[str]) -> str:
        """
        Normalizar campo gender para formato em inglês exigido pelo OASIS
        
        OASIS exige: male, female, other
        """
        if not gender:
            return "other"
        
        gender_lower = gender.lower().strip()
        
        # Mapeamento de chinês
        gender_map = {
            "男": "male",
            "女": "female",
            "机构": "other",
            "其他": "other",
            # Inglês já existente
            "male": "male",
            "female": "female",
            "other": "other",
        }
        
        return gender_map.get(gender_lower, "other")
    
    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Salvar Reddit Profile em formato JSON
        
        Usa formato consistente com to_reddit_format(), garantindo que o OASIS possa ler corretamente.
        Deve incluir o campo user_id, que é a chave para correspondência com agent_graph.get_agent() do OASIS!
        
        Campos obrigatórios:
        - user_id: ID do usuário (inteiro, usado para corresponder com poster_agent_id em initial_posts)
        - username: Nome de usuário
        - name: Nome de exibição
        - bio: Biografia
        - persona: Persona detalhada
        - age: Idade (inteiro)
        - gender: "male", "female", ou "other"
        - mbti: Tipo MBTI
        - country: País
        """
        data = []
        for idx, profile in enumerate(profiles):
            # Usar formato consistente com to_reddit_format()
            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,  # Chave: deve incluir user_id
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # Campos obrigatórios do OASIS - garantir valores padrão
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "中国",
            }
            
            # Campos opcionais
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics
            
            data.append(item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Salvos {len(profiles)} Reddit Profiles em {file_path} (formato JSON, com campo user_id)")
    
    # Manter nome do método antigo como alias, para compatibilidade retroativa
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """[Descontinuado] Por favor, use o método save_profiles()"""
        logger.warning("save_profiles_to_json está descontinuado, por favor use o método save_profiles")
        self.save_profiles(profiles, file_path, platform)

