# lunar-python Solar Term (节气) Audit Report

Rigorous verification of solar term boundaries for bāzì calculator engine.
Month pillar switches at the exact moment of 12 solar terms (節): 立春, 惊蛰, 清明, 立夏, 芒种, 小暑, 立秋, 白露, 寒露, 立冬, 大雪, 小寒.

---

## Task 1: Solar term times extracted from lunar-python

Precision: library returns `Solar` with year, month, day, hour, minute, second (date+time+minute).
Times are in **Beijing time (UTC+8)** per library convention.

### Year 1980

- **立春**: 1980-02-05 00:09:28
- **惊蛰**: 1980-03-05 18:16:29
- **清明**: 1980-04-04 23:14:42
- **立夏**: 1980-05-05 16:44:28
- **芒种**: 1980-06-05 21:03:44
- **小暑**: 1980-07-07 07:23:56
- **立秋**: 1980-08-07 17:08:30
- **白露**: 1980-09-07 19:53:27
- **寒露**: 1980-10-08 11:19:14
- **立冬**: 1980-11-07 14:18:13
- **大雪**: 1980-12-07 07:01:15
- **小寒**: 1980-01-06 12:28:53

### Year 1990

- **立春**: 1990-02-04 10:14:00
- **惊蛰**: 1990-03-06 04:19:18
- **清明**: 1990-04-05 09:12:56
- **立夏**: 1990-05-06 02:35:26
- **芒种**: 1990-06-06 06:46:18
- **小暑**: 1990-07-07 17:00:28
- **立秋**: 1990-08-08 02:45:32
- **白露**: 1990-09-08 05:37:28
- **寒露**: 1990-10-08 21:13:49
- **立冬**: 1990-11-08 00:23:30
- **大雪**: 1990-12-07 17:14:10
- **小寒**: 1990-01-05 22:33:14

### Year 2000

- **立春**: 2000-02-04 20:40:24
- **惊蛰**: 2000-03-05 14:42:40
- **清明**: 2000-04-04 19:31:58
- **立夏**: 2000-05-05 12:50:10
- **芒种**: 2000-06-05 16:58:34
- **小暑**: 2000-07-07 03:13:56
- **立秋**: 2000-08-07 13:02:59
- **白露**: 2000-09-07 15:59:10
- **寒露**: 2000-10-08 07:38:13
- **立冬**: 2000-11-07 10:48:04
- **大雪**: 2000-12-07 03:37:02
- **小寒**: 2000-01-06 09:00:42

### Year 2010

- **立春**: 2010-02-04 06:47:51
- **惊蛰**: 2010-03-06 00:46:22
- **清明**: 2010-04-05 05:30:30
- **立夏**: 2010-05-05 22:44:02
- **芒种**: 2010-06-06 02:49:24
- **小暑**: 2010-07-07 13:02:23
- **立秋**: 2010-08-07 22:49:07
- **白露**: 2010-09-08 01:44:41
- **寒露**: 2010-10-08 17:26:29
- **立冬**: 2010-11-07 20:42:30
- **大雪**: 2010-12-07 13:38:23
- **小寒**: 2010-01-05 19:08:47

### Year 2020

- **立春**: 2020-02-04 17:03:19
- **惊蛰**: 2020-03-05 10:56:52
- **清明**: 2020-04-04 15:38:09
- **立夏**: 2020-05-05 08:51:23
- **芒种**: 2020-06-05 12:58:26
- **小暑**: 2020-07-06 23:14:28
- **立秋**: 2020-08-07 09:06:11
- **白露**: 2020-09-07 12:08:02
- **寒露**: 2020-10-08 03:55:16
- **立冬**: 2020-11-07 07:13:55
- **大雪**: 2020-12-07 00:09:30
- **小寒**: 2020-01-06 05:30:06

---

## Task 2: Cross-reference with authoritative source (立春)

Reference (UTC+8): 立春 2024 Feb 4 16:27; 2023 Feb 4 10:42; 1990 Feb 4 10:18 (sources vary 10:14–10:18; ≤5 min treated as PASS).

- **立春 2024**: library 2024-02-04 16:27 vs ref 2024-02-04 16:27 (Δ 0 min). **PASS**

- **立春 2023**: library 2023-02-04 10:42 vs ref 2023-02-04 10:42 (Δ 0 min). **PASS**

- **立春 1990**: library 1990-02-04 10:14 vs ref 1990-02-04 10:18 (Δ -4 min). **PASS** (sources vary 10:14–10:18)

---

## Task 3: Month pillar flip at boundaries

### 立春 2024 (boundary 2024-02-04 16:27)
- Input: 1 min before / 1 min after boundary.
- Expected: before=丑月, after=寅月.
- Actual: before=丑月, after=寅月.
- Pillar flips: True; before match: True; after match: True.
- **PASS**

### 惊蛰 2024 (boundary 2024-03-05 10:22)
- Input: 1 min before / 1 min after boundary.
- Expected: before=寅月, after=卯月.
- Actual: before=寅月, after=卯月.
- Pillar flips: True; before match: True; after match: True.
- **PASS**

### 立春 1990 (boundary 1990-02-04 10:14)
- Input: 1 min before / 1 min after boundary.
- Expected: before=丑月, after=寅月.
- Actual: before=丑月, after=寅月.
- Pillar flips: True; before match: True; after match: True.
- **PASS**

---

## Task 4: 子 hour edge case (23:00–01:00 crossing midnight)

子时 before midnight → current day pillar; after midnight → next day pillar.

- Input: 1990-12-31 23:30.
  - Expected day pillar: 庚午.
  - Actual day pillar: 庚午.
  - **PASS**

- Input: 1991-01-01 00:30.
  - Expected day pillar: 辛未.
  - Actual day pillar: 辛未.
  - **PASS**

---

## Task 5: Year-boundary edge case (小寒 in early January)

- Test year: 2010; 小寒: 2010-01-05 19:08:47.
- Birth: Jan 3, 2010 12:00.
- Expected: 子月 (and correct previous 干支 year).
- Actual month pillar: 丙子; year pillar: 己丑.
- **PASS**

---
