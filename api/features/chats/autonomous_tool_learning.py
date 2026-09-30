"""
V121: Autonomous Tool Learning & Integration (ATLI)
自主工具学习与集成 - 自动学习并集成外部工具

创新点：
1. 工具发现与评估：自动发现可用工具
2. API自动调用：智能生成API调用
3. 工具链组合优化：组合多个工具解决问题
4. 工具效果反馈学习：从使用经验中学习
"""

import json
import re
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import random


class ToolType(Enum):
    """工具类型"""
    SEARCH = "search"
    CALCULATOR = "calculator"
    TRANSLATOR = "translator"
    CODE_EXECUTOR = "code_executor"
    DATABASE = "database"
    API_CLIENT = "api_client"
    FILE_READER = "file_reader"
    WEB_SCRAPER = "web_scraper"
    KNOWLEDGE_BASE = "knowledge_base"


class ToolCapability(Enum):
    """工具能力"""
    QUERY = "query"
    COMPUTE = "compute"
    TRANSFORM = "transform"
    RETRIEVE = "retrieve"
    EXECUTE = "execute"
    STORE = "store"


@dataclass
class Tool:
    """工具定义"""
    tool_id: str
    name: str
    tool_type: ToolType
    description: str
    parameters: Dict[str, Any]
    capabilities: List[ToolCapability]
    usage_count: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class ToolCall:
    """工具调用"""
    call_id: str
    tool_id: str
    parameters: Dict[str, Any]
    result: Any = None
    success: bool = False
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolChain:
    """工具链"""
    chain_id: str
    tools: List[str]  # tool_ids
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    use_count: int = 0


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.capability_index: Dict[ToolCapability, List[str]] = defaultdict(list)
        self.type_index: Dict[ToolType, List[str]] = defaultdict(list)
    
    def register_tool(self, tool: Tool) -> bool:
        """注册工具"""
        if tool.tool_id in self.tools:
            return False
        
        self.tools[tool.tool_id] = tool
        
        # 更新索引
        for capability in tool.capabilities:
            self.capability_index[capability].append(tool.tool_id)
        
        self.type_index[tool.tool_type].append(tool.tool_id)
        
        return True
    
    def discover_tools(
        self,
        required_capabilities: List[ToolCapability],
        min_success_rate: float = 0.5
    ) -> List[Tool]:
        """
        发现满足要求的工具
        
        Args:
            required_capabilities: 必需的能力
            min_success_rate: 最低成功率
        
        Returns:
            符合条件的工具列表
        """
        candidates = set()
        
        # 基于能力发现
        for capability in required_capabilities:
            candidates.update(self.capability_index[capability])
        
        # 过滤和排序
        qualified_tools = []
        for tool_id in candidates:
            tool = self.tools.get(tool_id)
            if tool and tool.success_rate >= min_success_rate:
                qualified_tools.append(tool)
        
        # 按成功率排序
        qualified_tools.sort(key=lambda t: t.success_rate, reverse=True)
        
        return qualified_tools
    
    def evaluate_tool(
        self,
        tool_id: str,
        test_cases: List[Dict]
    ) -> Dict[str, Any]:
        """评估工具性能"""
        tool = self.tools.get(tool_id)
        if not tool:
            return {'error': 'Tool not found'}
        
        results = []
        
        for test_case in test_cases:
            # 模拟执行
            success = random.random() > 0.2  # 80%成功率
            execution_time = random.uniform(0.1, 2.0)
            
            results.append({
                'test_case': test_case,
                'success': success,
                'execution_time': execution_time
            })
        
        # 计算指标
        success_count = sum(1 for r in results if r['success'])
        avg_time = sum(r['execution_time'] for r in results) / len(results)
        
        return {
            'tool_id': tool_id,
            'total_tests': len(test_cases),
            'success_count': success_count,
            'success_rate': success_count / len(test_cases),
            'average_execution_time': avg_time,
            'test_results': results
        }
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """获取工具统计信息"""
        if not self.tools:
            return {'total_tools': 0}
        
        return {
            'total_tools': len(self.tools),
            'by_type': {t.value: len(ids) for t, ids in self.type_index.items()},
            'by_capability': {c.value: len(ids) for c, ids in self.capability_index.items()},
            'top_tools': sorted(
                self.tools.values(),
                key=lambda t: t.success_rate,
                reverse=True
            )[:5]
        }


class APIGenerator:
    """API生成器"""
    
    def __init__(self):
        self.api_templates: Dict[str, str] = {
            'search': 'https://api.search.com/v1/query?q={query}&limit={limit}',
            'calculator': 'https://api.calc.com/v1/compute?expression={expression}',
            'translate': 'https://api.translate.com/v1/translate?text={text}&target={target}',
            'weather': 'https://api.weather.com/v1/current?location={location}'
        }
        self.call_history: List[Dict] = []
    
    def generate_api_call(
        self,
        tool: Tool,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成API调用
        
        Args:
            tool: 工具定义
            parameters: 调用参数
        
        Returns:
            API调用配置
        """
        # 1. 验证参数
        validation = self._validate_parameters(tool, parameters)
        if not validation['valid']:
            return {
                'success': False,
                'error': validation['error'],
                'missing_params': validation.get('missing', [])
            }
        
        # 2. 生成调用URL/命令
        api_config = self._build_api_config(tool, parameters)
        
        # 3. 生成示例代码
        example_code = self._generate_example_code(tool, parameters)
        
        call_config = {
            'tool_id': tool.tool_id,
            'tool_name': tool.name,
            'api_config': api_config,
            'example_code': example_code,
            'parameters': parameters,
            'expected_output': self._infer_output_schema(tool),
            'timestamp': datetime.now()
        }
        
        self.call_history.append(call_config)
        
        return {
            'success': True,
            'call_config': call_config
        }
    
    def _validate_parameters(
        self,
        tool: Tool,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证参数"""
        required_params = tool.parameters.get('required', [])
        provided_params = set(parameters.keys())
        
        missing = [p for p in required_params if p not in provided_params]
        
        if missing:
            return {
                'valid': False,
                'error': f'Missing required parameters: {missing}',
                'missing': missing
            }
        
        return {'valid': True}
    
    def _build_api_config(
        self,
        tool: Tool,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建API配置"""
        # 获取模板
        template = self.api_templates.get(tool.tool_type.value, '')
        
        # 填充参数
        url = template
        for key, value in parameters.items():
            placeholder = f'{{{key}}}'
            if placeholder in url:
                url = url.replace(placeholder, str(value))
        
        return {
            'method': 'GET' if tool.tool_type in [ToolType.SEARCH, ToolType.DATABASE] else 'POST',
            'url': url,
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer {API_KEY}'
            },
            'timeout': 30
        }
    
    def _generate_example_code(
        self,
        tool: Tool,
        parameters: Dict[str, Any]
    ) -> str:
        """生成示例代码"""
        code = f"""# 使用 {tool.name}
import requests

# API调用
response = requests.{self._get_http_method(tool)}(
    url="{self.api_templates.get(tool.tool_type.value, 'API_ENDPOINT')}",
    params={parameters},
    headers={{"Authorization": "Bearer YOUR_API_KEY"}}
)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print(f"Error: {{response.status_code}}")
"""
        return code
    
    def _get_http_method(self, tool: Tool) -> str:
        """获取HTTP方法"""
        if tool.tool_type in [ToolType.CALCULATOR, ToolType.CODE_EXECUTOR]:
            return 'post'
        return 'get'
    
    def _infer_output_schema(self, tool: Tool) -> Dict[str, Any]:
        """推断输出模式"""
        schemas = {
            ToolType.SEARCH: {'type': 'array', 'items': {'title': 'string', 'content': 'string'}},
            ToolType.CALCULATOR: {'type': 'number'},
            ToolType.TRANSLATOR: {'type': 'string'},
            ToolType.WEATHER: {'type': 'object', 'properties': {'temperature': 'number', 'condition': 'string'}}
        }
        
        return schemas.get(tool.tool_type, {'type': 'unknown'})


class ToolChainOptimizer:
    """工具链优化器"""
    
    def __init__(self):
        self.tool_chains: Dict[str, ToolChain] = {}
        self.execution_history: List[Dict] = []
    
    def compose_tool_chain(
        self,
        task_description: str,
        available_tools: List[Tool],
        max_chain_length: int = 5
    ) -> Dict[str, Any]:
        """
        组合工具链
        
        Args:
            task_description: 任务描述
            available_tools: 可用工具列表
            max_chain_length: 最大链长度
        
        Returns:
            工具链配置
        """
        # 1. 分析任务需求
        requirements = self._analyze_task_requirements(task_description)
        
        # 2. 规划工具序列
        tool_sequence = self._plan_tool_sequence(
            requirements,
            available_tools,
            max_chain_length
        )
        
        # 3. 验证工具链可行性
        validation = self._validate_tool_chain(tool_sequence)
        
        # 4. 生成数据流
        data_flow = self._generate_data_flow(tool_sequence)
        
        chain_id = f"chain_{len(self.tool_chains)}"
        
        tool_chain = ToolChain(
            chain_id=chain_id,
            tools=[t.tool_id for t in tool_sequence],
            input_schema=requirements['input_schema'],
            output_schema=self._infer_output_schema(tool_sequence)
        )
        
        self.tool_chains[chain_id] = tool_chain
        
        return {
            'chain_id': chain_id,
            'tool_sequence': [
                {'tool_id': t.tool_id, 'name': t.name, 'purpose': self._infer_tool_purpose(t, requirements)}
                for t in tool_sequence
            ],
            'data_flow': data_flow,
            'validation': validation,
            'estimated_execution_time': sum(t.avg_execution_time for t in tool_sequence)
        }
    
    def _analyze_task_requirements(self, task: str) -> Dict[str, Any]:
        """分析任务需求"""
        # 简化的需求分析
        requirements = {
            'capabilities_needed': [],
            'input_schema': {},
            'complexity': 'medium'
        }
        
        # 基于关键词识别需求
        if 'search' in task.lower() or 'find' in task.lower():
            requirements['capabilities_needed'].append(ToolCapability.RETRIEVE)
        
        if 'calculate' in task.lower() or 'compute' in task.lower():
            requirements['capabilities_needed'].append(ToolCapability.COMPUTE)
        
        if 'translate' in task.lower():
            requirements['capabilities_needed'].append(ToolCapability.TRANSFORM)
        
        return requirements
    
    def _plan_tool_sequence(
        self,
        requirements: Dict,
        available_tools: List[Tool],
        max_length: int
    ) -> List[Tool]:
        """规划工具序列"""
        sequence = []
        
        # 基于所需能力选择工具
        for capability in requirements['capabilities_needed']:
            matching_tools = [
                t for t in available_tools
                if capability in t.capabilities and t not in sequence
            ]
            
            if matching_tools:
                # 选择成功率最高的
                best_tool = max(matching_tools, key=lambda t: t.success_rate)
                sequence.append(best_tool)
            
            if len(sequence) >= max_length:
                break
        
        return sequence
    
    def _validate_tool_chain(self, sequence: List[Tool]) -> Dict[str, Any]:
        """验证工具链"""
        issues = []
        
        if len(sequence) == 0:
            issues.append("工具链为空")
        
        if len(sequence) > 5:
            issues.append("工具链过长，可能影响性能")
        
        # 检查工具兼容性（简化）
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_tool = sequence[i + 1]
            
            # 检查输出-输入兼容性
            if not self._are_tools_compatible(current, next_tool):
                issues.append(f"工具 {current.name} 和 {next_tool.name} 可能不兼容")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _are_tools_compatible(self, tool1: Tool, tool2: Tool) -> bool:
        """检查工具兼容性"""
        # 简化的兼容性检查
        return True
    
    def _generate_data_flow(self, sequence: List[Tool]) -> List[Dict]:
        """生成数据流"""
        flow = []
        
        for i, tool in enumerate(sequence):
            flow.append({
                'step': i + 1,
                'tool': tool.name,
                'input_from': 'previous_step' if i > 0 else 'user_input',
                'output_to': 'next_step' if i < len(sequence) - 1 else 'final_output'
            })
        
        return flow
    
    def _infer_tool_purpose(self, tool: Tool, requirements: Dict) -> str:
        """推断工具用途"""
        purposes = {
            ToolType.SEARCH: "检索相关信息",
            ToolType.CALCULATOR: "执行计算",
            ToolType.TRANSLATOR: "转换内容格式",
            ToolType.CODE_EXECUTOR: "执行代码",
            ToolType.DATABASE: "查询数据库"
        }
        
        return purposes.get(tool.tool_type, "处理数据")
    
    def _infer_output_schema(self, sequence: List[Tool]) -> Dict[str, Any]:
        """推断输出模式"""
        if not sequence:
            return {}
        
        last_tool = sequence[-1]
        
        schemas = {
            ToolType.SEARCH: {'results': 'array'},
            ToolType.CALCULATOR: {'result': 'number'},
            ToolType.TRANSLATOR: {'translated_text': 'string'}
        }
        
        return schemas.get(last_tool.tool_type, {'output': 'any'})
    
    def execute_tool_chain(
        self,
        chain_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行工具链"""
        chain = self.tool_chains.get(chain_id)
        if not chain:
            return {'error': 'Tool chain not found'}
        
        results = []
        current_data = input_data
        
        start_time = datetime.now()
        
        for tool_id in chain.tools:
            # 模拟工具执行
            execution_result = self._simulate_tool_execution(tool_id, current_data)
            results.append(execution_result)
            
            if not execution_result['success']:
                break
            
            current_data = execution_result['output']
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        execution_record = {
            'chain_id': chain_id,
            'input': input_data,
            'results': results,
            'execution_time': execution_time,
            'success': all(r['success'] for r in results),
            'timestamp': datetime.now()
        }
        
        self.execution_history.append(execution_record)
        
        # 更新工具链统计
        chain.use_count += 1
        chain.avg_execution_time = (
            (chain.avg_execution_time * (chain.use_count - 1) + execution_time)
            / chain.use_count
        )
        
        return execution_record
    
    def _simulate_tool_execution(
        self,
        tool_id: str,
        input_data: Dict
    ) -> Dict[str, Any]:
        """模拟工具执行"""
        # 模拟执行时间和成功率
        execution_time = random.uniform(0.1, 1.0)
        success = random.random() > 0.1  # 90%成功率
        
        return {
            'tool_id': tool_id,
            'input': input_data,
            'output': {'result': f'processed_{tool_id}'} if success else None,
            'success': success,
            'execution_time': execution_time
        }


class ToolLearningEngine:
    """工具学习引擎"""
    
    def __init__(self):
        self.usage_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.effectiveness_memory: Dict[str, Dict] = {}
        self.learning_history: List[Dict] = []
    
    def learn_from_usage(
        self,
        tool_id: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        从使用经验中学习
        
        Args:
            tool_id: 工具ID
            context: 使用上下文
            outcome: 使用结果
        
        Returns:
            学习结果
        """
        # 1. 记录使用模式
        usage_record = {
            'timestamp': datetime.now(),
            'context': context,
            'outcome': outcome
        }
        
        self.usage_patterns[tool_id].append(usage_record)
        
        # 2. 更新效果记忆
        if tool_id not in self.effectiveness_memory:
            self.effectiveness_memory[tool_id] = {
                'total_uses': 0,
                'successful_uses': 0,
                'contexts': defaultdict(int)
            }
        
        memory = self.effectiveness_memory[tool_id]
        memory['total_uses'] += 1
        
        if outcome.get('success', False):
            memory['successful_uses'] += 1
        
        # 记录上下文类型
        context_type = context.get('type', 'general')
        memory['contexts'][context_type] += 1
        
        # 3. 生成学习洞察
        insights = self._generate_learning_insights(tool_id)
        
        # 4. 更新工具参数建议
        recommendations = self._update_recommendations(tool_id)
        
        learning_result = {
            'tool_id': tool_id,
            'updated_memory': memory,
            'insights': insights,
            'recommendations': recommendations,
            'learning_progress': self._calculate_learning_progress(tool_id)
        }
        
        self.learning_history.append(learning_result)
        
        return learning_result
    
    def _generate_learning_insights(self, tool_id: str) -> List[Dict]:
        """生成学习洞察"""
        insights = []
        
        memory = self.effectiveness_memory.get(tool_id)
        if not memory:
            return insights
        
        # 成功率洞察
        success_rate = memory['successful_uses'] / max(memory['total_uses'], 1)
        
        if success_rate > 0.8:
            insights.append({
                'type': 'high_effectiveness',
                'description': f'工具 {tool_id} 在大多数场景下表现良好',
                'confidence': success_rate
            })
        elif success_rate < 0.5:
            insights.append({
                'type': 'low_effectiveness',
                'description': f'工具 {tool_id} 需要改进或限制使用场景',
                'confidence': 1 - success_rate
            })
        
        # 上下文偏好洞察
        if memory['contexts']:
            preferred_context = max(memory['contexts'].items(), key=lambda x: x[1])
            insights.append({
                'type': 'context_preference',
                'description': f'工具在 {preferred_context[0]} 场景下使用最频繁',
                'confidence': preferred_context[1] / memory['total_uses']
            })
        
        return insights
    
    def _update_recommendations(self, tool_id: str) -> List[Dict]:
        """更新推荐"""
        recommendations = []
        
        memory = self.effectiveness_memory.get(tool_id)
        if not memory:
            return recommendations
        
        success_rate = memory['successful_uses'] / max(memory['total_uses'], 1)
        
        if success_rate < 0.5:
            recommendations.append({
                'type': 'usage_restriction',
                'suggestion': '限制此工具的使用，仅在必要时调用',
                'priority': 'high'
            })
        
        if memory['total_uses'] > 10 and success_rate > 0.8:
            recommendations.append({
                'type': 'preferred_tool',
                'suggestion': '将此工具标记为首选工具',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _calculate_learning_progress(self, tool_id: str) -> Dict[str, Any]:
        """计算学习进度"""
        patterns = self.usage_patterns.get(tool_id, [])
        memory = self.effectiveness_memory.get(tool_id, {})
        
        return {
            'total_experiences': len(patterns),
            'success_rate': memory.get('successful_uses', 0) / max(memory.get('total_uses', 1), 1),
            'confidence_level': min(1.0, len(patterns) / 20),  # 20次使用后达到满信心
            'mastery_level': 'expert' if len(patterns) > 50 else 'intermediate' if len(patterns) > 10 else 'novice'
        }
    
    def predict_tool_effectiveness(
        self,
        tool_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """预测工具效果"""
        memory = self.effectiveness_memory.get(tool_id)
        
        if not memory or memory['total_uses'] < 5:
            return {
                'predicted_success_rate': 0.5,
                'confidence': 0.3,
                'recommendation': 'insufficient_data'
            }
        
        base_rate = memory['successful_uses'] / memory['total_uses']
        
        # 基于上下文调整
        context_type = context.get('type', 'general')
        context_success = self._get_context_success_rate(tool_id, context_type)
        
        if context_success > 0:
            predicted_rate = (base_rate + context_success) / 2
        else:
            predicted_rate = base_rate
        
        return {
            'predicted_success_rate': predicted_rate,
            'confidence': min(1.0, memory['total_uses'] / 20),
            'recommendation': 'use' if predicted_rate > 0.7 else 'caution' if predicted_rate > 0.4 else 'avoid'
        }
    
    def _get_context_success_rate(self, tool_id: str, context_type: str) -> float:
        """获取特定上下文的成功率"""
        patterns = self.usage_patterns.get(tool_id, [])
        
        context_patterns = [
            p for p in patterns
            if p['context'].get('type') == context_type
        ]
        
        if not context_patterns:
            return 0.0
        
        successful = sum(
            1 for p in context_patterns
            if p['outcome'].get('success', False)
        )
        
        return successful / len(context_patterns)


# 主引擎类
class AutonomousToolLearningEngine:
    """自主工具学习引擎"""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.api_generator = APIGenerator()
        self.chain_optimizer = ToolChainOptimizer()
        self.learning_engine = ToolLearningEngine()
        self._init_default_tools()
    
    def _init_default_tools(self):
        """初始化默认工具"""
        default_tools = [
            Tool(
                tool_id='search_tool',
                name='Web Search',
                tool_type=ToolType.SEARCH,
                description='Search the web for information',
                parameters={'required': ['query'], 'optional': ['limit']},
                capabilities=[ToolCapability.RETRIEVE, ToolCapability.QUERY],
                success_rate=0.85
            ),
            Tool(
                tool_id='calc_tool',
                name='Calculator',
                tool_type=ToolType.CALCULATOR,
                description='Perform mathematical calculations',
                parameters={'required': ['expression']},
                capabilities=[ToolCapability.COMPUTE],
                success_rate=0.95
            ),
            Tool(
                tool_id='translate_tool',
                name='Translator',
                tool_type=ToolType.TRANSLATOR,
                description='Translate text between languages',
                parameters={'required': ['text', 'target']},
                capabilities=[ToolCapability.TRANSFORM],
                success_rate=0.80
            )
        ]
        
        for tool in default_tools:
            self.tool_registry.register_tool(tool)
    
    def process_with_tools(
        self,
        task: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用工具处理任务
        
        这是V121创新的主要入口函数
        """
        # 1. 发现合适的工具
        requirements = self._extract_requirements(task)
        discovered_tools = self.tool_registry.discover_tools(
            requirements['capabilities']
        )
        
        # 2. 组合工具链
        chain_result = self.chain_optimizer.compose_tool_chain(
            task,
            discovered_tools
        )
        
        # 3. 执行工具链
        if chain_result.get('validation', {}).get('valid', False):
            execution_result = self.chain_optimizer.execute_tool_chain(
                chain_result['chain_id'],
                input_data
            )
        else:
            execution_result = {'error': 'Invalid tool chain', 'success': False}
        
        # 4. 从执行结果中学习
        if execution_result.get('success'):
            for tool_id in chain_result.get('tool_sequence', []):
                self.learning_engine.learn_from_usage(
                    tool_id['tool_id'],
                    {'type': task, 'input': input_data},
                    {'success': True, 'output': execution_result.get('results')}
                )
        
        # 5. 生成API调用示例
        api_examples = []
        for tool_info in chain_result.get('tool_sequence', []):
            tool = self.tool_registry.tools.get(tool_info['tool_id'])
            if tool:
                api_result = self.api_generator.generate_api_call(
                    tool,
                    input_data
                )
                if api_result['success']:
                    api_examples.append(api_result['call_config'])
        
        return {
            'version': 'V121',
            'name': 'Autonomous Tool Learning & Integration (ATLI)',
            'task': task,
            'discovered_tools': [
                {'tool_id': t.tool_id, 'name': t.name, 'success_rate': t.success_rate}
                for t in discovered_tools[:5]
            ],
            'tool_chain': chain_result,
            'execution_result': execution_result,
            'api_examples': api_examples,
            'learning_insights': self._get_learning_summary(),
            'recommendations': self._generate_recommendations(execution_result)
        }
    
    def _extract_requirements(self, task: str) -> Dict[str, Any]:
        """提取任务需求"""
        capabilities = []
        
        if any(word in task.lower() for word in ['search', 'find', 'look up', '查询']):
            capabilities.append(ToolCapability.RETRIEVE)
        
        if any(word in task.lower() for word in ['calculate', 'compute', '计算']):
            capabilities.append(ToolCapability.COMPUTE)
        
        if any(word in task.lower() for word in ['translate', 'convert', '转换']):
            capabilities.append(ToolCapability.TRANSFORM)
        
        if not capabilities:
            capabilities = [ToolCapability.QUERY]
        
        return {'capabilities': capabilities}
    
    def _get_learning_summary(self) -> Dict[str, Any]:
        """获取学习摘要"""
        return {
            'total_tools_learned': len(self.learning_engine.effectiveness_memory),
            'total_experiences': sum(
                len(patterns) for patterns in self.learning_engine.usage_patterns.values()
            ),
            'tool_statistics': self.tool_registry.get_tool_statistics()
        }
    
    def _generate_recommendations(self, execution_result: Dict) -> List[str]:
        """生成推荐"""
        recommendations = []
        
        if execution_result.get('success'):
            recommendations.append("工具链执行成功，可以保存此配置供将来使用")
        else:
            recommendations.append("工具链执行失败，建议检查工具兼容性或尝试替代工具")
        
        return recommendations


# 便捷函数
def build_autonomous_tool_learning(
    task: str,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    构建自主工具学习
    
    这是V121创新的便捷入口函数
    """
    engine = AutonomousToolLearningEngine()
    return engine.process_with_tools(task, input_data)


# 示例用法
if __name__ == "__main__":
    result = build_autonomous_tool_learning(
        "Search for machine learning papers and calculate the average citation count",
        {'query': 'machine learning', 'field': 'computer science'}
    )
    
    print(f"Task: {result['task']}")
    print(f"Tools discovered: {len(result['discovered_tools'])}")
    print(f"Chain validation: {result['tool_chain'].get('validation', {})}")
