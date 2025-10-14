# agent_card.py
import os
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

k8s_skill = AgentSkill(
    id="k8s_query",
    name="Query Kubernetes",
    description="Allows you to request information, metrics, and status from your Kubernetes cluster using MCP protocol.",
    tags=["k8s", "monitoring", "metrics", "cluster", "pods", "nodes", "alerts", "resources"],
    examples=[
        "Show me active alerts",
        "What is the CPU usage for node 'worker-1'?",
        "List all running pods in namespace 'production'",
        "How much memory is used by the cluster?",
        "Show me the status of deployments",
        "Which pods are in CrashLoopBackOff?",
        "Get the last 10 events for pod 'api-server'",
        "Show me the resource usage trend for the last 24 hours"
    ],
)

scale_skill = AgentSkill(
    id="scale_resource",
    name="Scale Resources",
    description="Scale deployments, statefulsets, or other resources in your Kubernetes cluster.",
    tags=["k8s", "scaling", "deployments", "statefulsets", "resources"],
    examples=[
        "Scale deployment 'web-app' to 5 replicas",
        "Increase memory for statefulset 'db-cluster'",
        "Reduce CPU requests for deployment 'api-server'"
    ],
)

troubleshoot_skill = AgentSkill(
    id="troubleshoot",
    name="Troubleshoot Issues",
    description="Diagnose and suggest solutions for common Kubernetes problems.",
    tags=["k8s", "troubleshooting", "diagnostics", "alerts", "errors"],
    examples=[
        "Why is pod 'worker-2' in CrashLoopBackOff?",
        "How can I fix image pull errors?",
        "Show me failed jobs in the last hour"
    ],
)

public_agent_card = AgentCard(
    name="Kubernetes AI Agent",
    description=(
        "An advanced AI agent for Kubernetes that answers questions, provides metrics, troubleshooting, and automation via MCP protocol. "
        "Supports natural language queries, streaming responses, and multiple skills for cluster management."
    ),
    url=os.getenv("AGENT_URL", "http://localhost:8080/"),
    version="1.0.0",
    author="OpenBear IT",
    documentation_url="https://github.com/openbear-it/ai-kubernetes-agent",
    contact_email="support@openbear.it",
    license="Apache-2.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True, authenticated=True, supports_scaling=True),
    skills=[k8s_skill, scale_skill, troubleshoot_skill],
    supports_authenticated_extended_card=True,
    tags=["kubernetes", "ai", "monitoring", "automation", "MCP", "cloud", "devops"],
    examples=[
        "Show me all pods in namespace 'dev'",
        "Scale deployment 'api-server' to 10 replicas",
        "Why is my pod not starting?",
        "List all nodes and their status",
        "Get CPU and memory usage for the last 6 hours"
    ]
)
