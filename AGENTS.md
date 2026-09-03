# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Chạy code trên server IBM

- KHÔNG train/chạy code nặng trên máy local. Toàn bộ training/eval chạy trên IBM qua SSH.
- SSH wrapper: `bash ~/ssh_ibm.sh "lệnh"` (đã cấu hình key + user ubuntu@128.168.135.246).
- Môi trường: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310` (torch cu130, GPU L40S).
- Thư mục code trên IBM: `/home/ubuntu/woker/Zeroshot/HSPRM` — luôn `cd` vào trước khi chạy.
- Sync file từ WSL lên IBM:
  `/mnt/c/Windows/System32/OpenSSH/scp.exe -o BatchMode=yes -o IdentitiesOnly=yes -i 'D:\Work\ssh\tongdat_rsa.prv' <file_local> ubuntu@128.168.135.246:<đường_dẫn_trên_IBM>`
  rồi chạy `sed -i 's/\r$//' <file>` trên IBM nếu file chứa CRLF.
- Train dài: dùng nohup + ghi log, kiểm tra bằng `tail log`. VD:
  `nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310 && CUDA_VISIBLE_DEVICES=0 python train.py --config config/<cfg>.yaml' > logs/<run>.log 2>&1 &`
- Lệnh ngắn (<2 phút) chạy trực tiếp; lệnh dài luôn background + log.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.