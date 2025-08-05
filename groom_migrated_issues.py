#!/usr/bin/env python3
"""
Comprehensive grooming script for the 18 migrated GitHub issues.
Adds proper priority, component, and status labels for better organization.
"""

import json
import subprocess
import sys
from typing import Dict, List, Tuple


def run_gh_command(cmd: List[str]) -> str:
    """Run a GitHub CLI command and return the output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}: {e.stderr}")
        return ""


def get_migrated_issues() -> List[Dict]:
    """Get all migrated issues with their current labels."""
    cmd = ["gh", "issue", "list", "--label", "migrated-from-yaml", "--state", "open", 
           "--json", "number,title,labels,body"]
    result = run_gh_command(cmd)
    if result:
        return json.loads(result)
    return []


def analyze_issue_complexity(title: str, body: str) -> str:
    """Analyze issue complexity based on title and description."""
    complex_keywords = [
        "refactor", "architecture", "comprehensive", "framework", 
        "scalable", "centralized", "predictive", "fine-tune"
    ]
    medium_keywords = [
        "implement", "improve", "enhance", "consolidate", 
        "establish", "structured", "logging"
    ]
    
    text = (title + " " + body).lower()
    
    if any(keyword in text for keyword in complex_keywords):
        return "high"
    elif any(keyword in text for keyword in medium_keywords):
        return "medium"
    else:
        return "low"


def determine_component(title: str, body: str, current_labels: List[str]) -> str:
    """Determine the primary component based on issue content."""
    text = (title + " " + body).lower()
    
    # Check existing labels first
    for label in current_labels:
        if label in ["backend", "frontend", "cli", "data", "ai", "ml"]:
            return label
    
    # Analyze content
    if any(word in text for word in ["api", "server", "backend", "orchestrator", "gatekeeper"]):
        return "backend"
    elif any(word in text for word in ["ui", "frontend", "interface", "user"]):
        return "frontend"
    elif any(word in text for word in ["cli", "command", "typer"]):
        return "cli"
    elif any(word in text for word in ["test", "testing", "pytest"]):
        return "testing"
    elif any(word in text for word in ["doc", "documentation", "sphinx"]):
        return "docs"
    elif any(word in text for word in ["ai", "llm", "model", "fine-tune"]):
        return "ai"
    elif any(word in text for word in ["data", "database", "storage"]):
        return "data"
    else:
        return "backend"  # default


def determine_priority(title: str, body: str, current_labels: List[str]) -> str:
    """Determine issue priority."""
    text = (title + " " + body).lower()
    
    # High priority items
    if any(word in text for word in ["security", "critical", "urgent", "dependencies", "audit"]):
        return "high"
    
    # Phase-based priority
    if any(label.startswith("phase-") for label in current_labels):
        phase_num = None
        for label in current_labels:
            if label.startswith("phase-"):
                phase_num = int(label.split("-")[1])
                break
        
        if phase_num is not None:
            if phase_num <= 1:
                return "high"
            elif phase_num == 2:
                return "medium" 
            else:
                return "low"
    
    # Content-based priority
    if any(word in text for word in ["comprehensive", "framework", "refactor"]):
        return "medium"
    
    return "low"


def determine_effort(title: str, body: str) -> str:
    """Estimate effort required."""
    text = (title + " " + body).lower()
    
    large_effort = [
        "comprehensive", "framework", "refactor", "consolidate",
        "fine-tune", "scalable", "centralized"
    ]
    medium_effort = [
        "implement", "improve", "establish", "enhance", "update"
    ]
    
    if any(word in text for word in large_effort):
        return "large"
    elif any(word in text for word in medium_effort):
        return "medium"
    else:
        return "small"


def groom_issue(issue: Dict, dry_run: bool = False) -> None:
    """Apply comprehensive grooming to a single issue."""
    number = issue["number"]
    title = issue["title"]
    body = issue.get("body", "")
    current_labels = [label["name"] for label in issue["labels"]]
    
    print(f"\n🔍 Grooming Issue #{number}: {title}")
    
    # Analyze issue
    priority = determine_priority(title, body, current_labels)
    component = determine_component(title, body, current_labels)
    effort = determine_effort(title, body)
    complexity = analyze_issue_complexity(title, body)
    
    # Determine status based on content
    status = "todo"  # Default for migrated issues
    if any(word in title.lower() for word in ["security", "dependencies", "audit"]):
        status = "ready"  # High priority items are ready to start
    
    # Build new labels to add
    new_labels = []
    
    # Priority labels
    if not any(label.endswith("-priority") for label in current_labels):
        new_labels.append(f"{priority}-priority")
    
    # Component labels (ensure primary component is set)
    if not any(label in current_labels for label in ["backend", "frontend", "cli", "data", "ai", "ml", "docs", "testing"]):
        new_labels.append(component)
    
    # Effort estimation
    if not any(label.endswith("-effort") for label in current_labels):
        new_labels.append(f"{effort}-effort")
    
    # Status labels
    if not any(label in current_labels for label in ["todo", "ready", "in-progress", "blocked"]):
        new_labels.append(status)
    
    # Type classification
    type_labels = ["feature", "enhancement", "bug", "refactor", "documentation"]
    if not any(label in current_labels for label in type_labels):
        if "refactor" in current_labels:
            pass  # Already has refactor
        elif any(word in title.lower() for word in ["doc", "documentation"]):
            new_labels.append("documentation")
        elif any(word in title.lower() for word in ["enhance", "improve"]):
            new_labels.append("enhancement")
        else:
            new_labels.append("feature")
    
    # Clean up numeric labels (1, 2, 3, 4, 5)
    labels_to_remove = [label for label in current_labels if label.isdigit()]
    
    print(f"  📊 Analysis:")
    print(f"    Priority: {priority}")
    print(f"    Component: {component}")
    print(f"    Effort: {effort}")
    print(f"    Status: {status}")
    print(f"    Complexity: {complexity}")
    
    if new_labels:
        print(f"  ➕ Adding labels: {', '.join(new_labels)}")
        if not dry_run:
            cmd = ["gh", "issue", "edit", str(number), "--add-label", ",".join(new_labels)]
            run_gh_command(cmd)
    
    if labels_to_remove:
        print(f"  ➖ Removing labels: {', '.join(labels_to_remove)}")
        if not dry_run:
            cmd = ["gh", "issue", "edit", str(number), "--remove-label", ",".join(labels_to_remove)]
            run_gh_command(cmd)
    
    # Add assignee suggestions based on component
    assignee_map = {
        "backend": "adrianwedd",
        "frontend": "adrianwedd", 
        "ai": "adrianwedd",
        "security": "adrianwedd",
        "docs": "adrianwedd",
        "testing": "adrianwedd",
    }
    
    suggested_assignee = assignee_map.get(component)
    if suggested_assignee:
        print(f"  👤 Suggested assignee: {suggested_assignee}")
        # Note: Not auto-assigning to avoid overwhelming


def create_grooming_summary(issues: List[Dict]) -> None:
    """Create a summary of grooming results."""
    print("\n" + "="*60)
    print("📋 GROOMING SUMMARY")
    print("="*60)
    
    priorities = {"high": 0, "medium": 0, "low": 0}
    components = {}
    efforts = {"small": 0, "medium": 0, "large": 0}
    
    for issue in issues:
        title = issue["title"]
        body = issue.get("body", "")
        current_labels = [label["name"] for label in issue["labels"]]
        
        priority = determine_priority(title, body, current_labels)
        component = determine_component(title, body, current_labels)
        effort = determine_effort(title, body)
        
        priorities[priority] += 1
        components[component] = components.get(component, 0) + 1
        efforts[effort] += 1
    
    print("\n📊 Priority Distribution:")
    for priority, count in priorities.items():
        print(f"  {priority.capitalize()}: {count} issues")
    
    print("\n🔧 Component Distribution:")
    for component, count in sorted(components.items()):
        print(f"  {component.capitalize()}: {count} issues")
    
    print("\n⏱️ Effort Distribution:")
    for effort, count in efforts.items():
        print(f"  {effort.capitalize()}: {count} issues")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("1. Start with high-priority security and dependency issues")
    print("2. Focus on backend issues first (likely foundational)")
    print("3. Group small-effort items for quick wins")
    print("4. Plan large-effort items for dedicated sprints")
    
    # Quick commands for next steps
    print("\n🚀 NEXT STEPS - Quick Commands:")
    print("# View high priority issues:")
    print("gh issue list --label 'high-priority,migrated-from-yaml' --state open")
    print("\n# View ready-to-start issues:")
    print("gh issue list --label 'ready,migrated-from-yaml' --state open")
    print("\n# Assign yourself to high priority items:")
    print("gh issue list --label 'high-priority,migrated-from-yaml' --state open --json number --jq '.[].number' | head -3 | xargs -I {} gh issue edit {} --add-assignee '@me'")


def main():
    """Main grooming function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Groom migrated GitHub issues")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()
    
    print("🧹 COMPREHENSIVE ISSUE GROOMING")
    print("===============================")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE UPDATE'}")
    
    # Get all migrated issues
    issues = get_migrated_issues()
    if not issues:
        print("❌ No migrated issues found!")
        return
    
    print(f"\n📌 Found {len(issues)} migrated issues to groom")
    
    # Groom each issue
    for issue in issues:
        groom_issue(issue, args.dry_run)
    
    # Create summary
    create_grooming_summary(issues)
    
    if args.dry_run:
        print("\n⚠️  This was a dry run. Add --no-dry-run to apply changes.")
    else:
        print("\n✅ Grooming completed!")


if __name__ == "__main__":
    main()