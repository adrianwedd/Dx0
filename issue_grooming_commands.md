# 🧹 Issue Grooming Commands - Dx0 Project

## ✅ Grooming Results Summary

**18 migrated issues successfully groomed with:**
- ✅ **Priority labels**: high-priority (3), medium-priority (7), low-priority (8)
- ✅ **Effort estimates**: small-effort (3), medium-effort (4), large-effort (11)
- ✅ **Status labels**: ready (1), todo (17)
- ✅ **Component labels**: backend (9), frontend (6), ai (1), cli (1), testing (1)
- ✅ **Cleaned up**: Removed numeric labels (1, 2, 3, 4, 5)

---

## 🚀 Daily Workflow Commands

### Morning Standup
```bash
# View high-priority items to tackle today
gh issue list --label 'high-priority,migrated-from-yaml' --state open

# Check what's ready to start immediately
gh issue list --label 'ready,migrated-from-yaml' --state open

# Quick wins (small effort items)
gh issue list --label 'small-effort,migrated-from-yaml' --state open
```

### Priority-Based Work Planning
```bash
# Focus areas by priority
gh issue list --label 'high-priority' --state open --json number,title,labels | jq -r '.[] | "# \(.number): \(.title)"'
gh issue list --label 'medium-priority' --state open --json number,title,labels | jq -r '.[] | "# \(.number): \(.title)"'

# Group by effort for sprint planning
gh issue list --label 'small-effort,migrated-from-yaml' --state open
gh issue list --label 'medium-effort,migrated-from-yaml' --state open  
gh issue list --label 'large-effort,migrated-from-yaml' --state open
```

### Component-Based Assignment
```bash
# Backend-focused work (9 issues)
gh issue list --label 'backend,migrated-from-yaml' --state open

# Frontend work (6 issues)  
gh issue list --label 'frontend,migrated-from-yaml' --state open

# Security-focused work
gh issue list --label 'security,migrated-from-yaml' --state open
```

---

## 🎯 Issue Management Commands

### Self-Assignment & Status Updates
```bash
# Assign high-priority items to yourself
gh issue list --label 'high-priority,migrated-from-yaml' --state open --json number --jq '.[].number' | head -3 | xargs -I {} gh issue edit {} --add-assignee '@me'

# Start working on an issue (change status)
gh issue edit ISSUE_NUMBER --remove-label "todo" --add-label "in-progress"

# Mark issue blocked (with reason)
gh issue edit ISSUE_NUMBER --remove-label "todo,in-progress" --add-label "blocked" --comment "Blocked by: [reason]"

# Complete an issue  
gh issue close ISSUE_NUMBER --comment "✅ Completed: [summary of work done]"
```

### Quick Status Transitions
```bash
# Move ready items to todo
gh issue list --label 'ready,migrated-from-yaml' --state open --json number --jq '.[].number' | xargs -I {} gh issue edit {} --remove-label "ready" --add-label "todo"

# Bulk start small-effort items
gh issue list --label 'small-effort,todo,migrated-from-yaml' --state open --json number --jq '.[].number' | head -2 | xargs -I {} gh issue edit {} --remove-label "todo" --add-label "in-progress" --add-assignee "@me"
```

---

## 📊 Analytics & Reporting

### Work Distribution Analysis
```bash
# Priority breakdown
echo "📊 Priority Distribution:"
echo "High: $(gh issue list --label 'high-priority,migrated-from-yaml' --state open | wc -l)"
echo "Medium: $(gh issue list --label 'medium-priority,migrated-from-yaml' --state open | wc -l)" 
echo "Low: $(gh issue list --label 'low-priority,migrated-from-yaml' --state open | wc -l)"

# Effort analysis
echo "⏱️ Effort Distribution:"  
echo "Small: $(gh issue list --label 'small-effort,migrated-from-yaml' --state open | wc -l)"
echo "Medium: $(gh issue list --label 'medium-effort,migrated-from-yaml' --state open | wc -l)"
echo "Large: $(gh issue list --label 'large-effort,migrated-from-yaml' --state open | wc -l)"

# Component workload
echo "🔧 Component Distribution:"
echo "Backend: $(gh issue list --label 'backend,migrated-from-yaml' --state open | wc -l)"
echo "Frontend: $(gh issue list --label 'frontend,migrated-from-yaml' --state open | wc -l)"
echo "AI/ML: $(gh issue list --label 'ai,migrated-from-yaml' --state open | wc -l)"
```

### Progress Tracking
```bash
# Overall progress on migrated issues
total=$(gh issue list --label 'migrated-from-yaml' | wc -l)
closed=$(gh issue list --label 'migrated-from-yaml' --state closed | wc -l)
echo "🎯 Migration Progress: $closed/$total completed ($(( closed * 100 / total ))%)"

# This week's velocity
gh issue list --label 'migrated-from-yaml' --state closed --json closedAt | jq --arg week "$(date -d '7 days ago' -I)" '.[] | select(.closedAt > $week)' | wc -l
```

---

## 🎯 Recommended Work Order

### Phase 1: Quick Security Wins (Start Here!)
```bash
# Issue #306: Pin and Audit Dependencies (READY - Small Effort)
gh issue view 306
gh issue edit 306 --add-assignee '@me'

# Issue #309: Harden XML Parsing (High Priority - Small Effort)  
gh issue view 309
gh issue edit 309 --add-assignee '@me'
```

### Phase 2: Foundation Building (Medium Priority, Large Impact)
```bash
# Testing Infrastructure
gh issue view 307  # Comprehensive Test Suite

# Configuration Management  
gh issue view 308  # Centralized Configuration

# Session Management
gh issue view 310  # Scalable Session Management
```

### Phase 3: Feature Development (After Foundation)
```bash
# Frontend Consolidation
gh issue view 313  # Single Modern UI

# API Improvements
gh issue view 312  # OpenAI API Update

# Documentation
gh issue view 315  # API Documentation
```

---

## 🛠️ Advanced Grooming Operations

### Dependency Chain Management
```bash
# Find issues that might block others
gh issue list --label 'backend,high-priority,migrated-from-yaml' --state open

# Group related issues
gh issue list --query "configuration OR session OR logging" --label 'migrated-from-yaml' --state open
```

### Sprint Planning Helper
```bash
# Small sprint (1-2 weeks): Focus on small-effort items
gh issue list --label 'small-effort,high-priority,migrated-from-yaml' --state open

# Medium sprint (2-4 weeks): Mix of small and medium effort
gh issue list --label 'small-effort,medium-effort,high-priority,migrated-from-yaml' --state open

# Large epic planning: Large effort items
gh issue list --label 'large-effort,migrated-from-yaml' --state open --json number,title,labels | jq -r '.[] | "Epic: #\(.number) - \(.title)"'
```

### Quality Gates
```bash
# Before marking large items as done, ensure dependencies
gh issue list --query "test OR documentation" --label 'migrated-from-yaml' --state open

# Security review checklist
gh issue list --label 'security,migrated-from-yaml' --state open
```

---

## 📋 Quick Reference

### Most Useful Daily Commands
```bash
# 🌅 Morning: What should I work on?
gh issue list --label 'high-priority,ready,migrated-from-yaml' --state open

# ⚡ Quick wins available?  
gh issue list --label 'small-effort,todo,migrated-from-yaml' --state open

# 🎯 My current assignments
gh issue list --assignee '@me' --label 'migrated-from-yaml' --state open

# 📈 Progress check
echo "Completed: $(gh issue list --label 'migrated-from-yaml' --state closed | wc -l)/18"
```

### Emergency Commands
```bash
# 🚨 Find critical security issues
gh issue list --label 'security,high-priority' --state open

# 🏃‍♂️ Find items ready to start NOW
gh issue list --label 'ready,small-effort' --state open

# 🔥 Show what's currently in progress
gh issue list --label 'in-progress,migrated-from-yaml' --state open
```

---

**🎉 Your 18 migrated issues are now fully groomed and ready for systematic execution!**

**Start with the high-priority, small-effort items (#306, #309) for quick wins, then tackle the foundational medium-priority items.**