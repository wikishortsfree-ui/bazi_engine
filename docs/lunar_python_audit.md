# lunar-python EightChar API Audit

**Test chart:** `Solar.fromYmdHms(2003, 4, 24, 21, 20, 0)` → 癸未 丙辰 丁卯 辛亥 (day master 丁).

---

## 1. Feature → method mapping

| Feature | Library method | Return value (test chart) | Status |
|--------|----------------|---------------------------|--------|
| **Four natal pillars (stem+branch)** | `getYear()`, `getMonth()`, `getDay()`, `getTime()` | `'癸未'`, `'丙辰'`, `'丁卯'`, `'辛亥'` | ✅ Full |
| **Five elements per pillar** | `getYearWuXing()`, `getMonthWuXing()`, `getDayWuXing()`, `getTimeWuXing()` | `'水土'`, `'火土'`, `'火木'`, `'金水'` (gan+zhi concatenated) | ✅ Full |
| **Ten gods – heavenly stems (vs day master)** | `getYearShiShenGan()`, `getMonthShiShenGan()`, `getDayShiShenGan()`, `getTimeShiShenGan()` | `'七杀'`, `'劫财'`, `'日主'`, `'偏财'` | ✅ Full |
| **Ten gods – earthly branches** | `getYearShiShenZhi()`, `getMonthShiShenZhi()`, `getDayShiShenZhi()`, `getTimeShiShenZhi()` | `['食神','比肩','偏印']`, `['伤官','偏印','七杀']`, `['偏印']`, `['正官','正印']` (lists by 藏干) | ✅ Full |
| **Hidden stems (藏干)** | `getYearHideGan()`, `getMonthHideGan()`, `getDayHideGan()`, `getTimeHideGan()` | `['己','丁','乙']`, `['戊','乙','癸']`, `['乙']`, `['壬','甲']` | ⚠️ Partial (no weights) |
| **大运 start age, stem+branch per period** | `ec.getYun(gender)` → `yun.getStartYear()`, `yun.getDaYun()`; `dy.getGanZhi()`, `dy.getStartYear()`, `dy.getStartAge()`, `dy.getEndYear()` | Start age 6; DaYun[0].getGanZhi()=`''`, DaYun[1]=`'乙卯'`, start year 2009, start age 7 | ✅ Full |
| **流年 stem+branch for a year** | `yun.getDaYun()[i].getLiuNian()` → list of LiuNian; `ln.getYear()`, `ln.getGanZhi()` | LiuNian[0].getYear()=2003, getGanZhi()=`'癸未'` | ⚠️ Partial (no direct getLiuNian(year)) |
| **流月 stem+branch for a month** | `ln.getLiuYue()` → list of LiuYue; `ly.getGanZhi()`, `ly.getMonthInChinese()` | 12 items; LiuYue[0].getGanZhi()=`'甲寅'`, getMonthInChinese()=`'正'` | ⚠️ Partial (index = lunar month; no direct getLiuYue(year, month)) |
| **Solar term (节气) boundaries** | `lunar.getJieQiTable()` → dict[name]=Solar; or `lunar.getPrevJieQi()`/`getNextJieQi()` → JieQi.getSolar()` | Prev 谷雨: Solar 2003-4-20 **20:02:48**; table values are Solar (fromJulianDay) | ✅ Full |

---

## 2. Hidden stems (藏干) – weights

- **What the library has:** `LunarUtil.ZHI_HIDE_GAN` (in `util/LunarUtil.py`): dict from 地支 to **list** of stems, order = 主气、余气、杂气. Example: `"辰": ["戊", "乙", "癸"]`.
- **What it returns:** `getYearHideGan()` etc. return that list only (e.g. `['戊','乙','癸']` for 辰). **No weights** (e.g. 戊60%, 乙30%, 癸10%) are exposed.
- **Conclusion:** 🔨 **Must build** a weight table yourself if you need 藏干 weights. You can reuse the same stem lists from the library (or copy `ZHI_HIDE_GAN`) and add your own weight mapping; the library does not expose or use weights.

---

## 3. EightChar methods NOT in your feature list

| Method | Return (test chart) | Usefulness |
|--------|----------------------|------------|
| `getYearNaYin()` / getMonthNaYin / getDayNaYin / getTimeNaYin | `'杨柳木'`, `'沙中土'`, `'炉中火'`, `'钗钏金'` | 纳音 – useful if you need 纳音. |
| `getYearXun()` / getMonthXun / getDayXun / getTimeXun | `'甲戌'`, `'甲寅'`, `'甲子'`, `'甲辰'` | 旬 – useful for 旬空. |
| `getYearXunKong()` / … / getTimeXunKong | `'申酉'`, `'子丑'`, `'戌亥'`, `'寅卯'` | 旬空 – useful. |
| `getYearDiShi()` / getMonthDiShi / getDayDiShi / getTimeDiShi | `'冠带'`, `'衰'`, `'病'`, `'胎'` | 十二长生 (di Shi) – useful for 大运/流年 strength. |
| `getMingGong()`, `getMingGongNaYin()` | `'甲寅'`, `'大溪水'` | 命宫 – useful if you use 命宫. |
| `getShenGong()`, `getShenGongNaYin()` | `'丙辰'`, `'沙中土'` | 身宫 – ditto. |
| `getTaiYuan()`, `getTaiYuanNaYin()` | `'丁未'`, `'天河水'` | 胎元 – ditto. |
| `getTaiXi()`, `getTaiXiNaYin()` | `'壬戌'`, `'大海水'` | 胎息 – ditto. |
| `getSect()`, `setSect(sect)` | `2` | 流派 (1=阴历, 2=盲派 etc.) – affects day master calculation. |
| `CHANG_SHENG`, `MONTH_ZHI` | Constants | 长生表; 月支序. |
| `getLunar()`, `fromLunar()`, `toString()` | Lunar obj; `'癸未 丙辰 丁卯 辛亥'` | Infrastructure. |

**Verdict:** 纳音、旬/旬空、十二长生、命宫/身宫/胎元/胎息 are all useful extras; sect is important for consistency with 盲派/阴历.

---

## 4. Gaps to build on top

1. **藏干 weights** – Define and use your own 主气/余气/杂气 weight table; library only gives stem lists.
2. **流年 by (calendar) year** – No `getLiuNian(2010)`. You must: find which `DaYun` contains the year (`dy.getStartYear()`/`getEndYear()`), then index into `dy.getLiuNian()` by year offset.
3. **流月 by (year, month)** – No direct API. Get the correct `LiuNian` for the year, then `getLiuYue()[lunar_month_index]` (正月=0, etc.); lunar month may need 节气 boundary logic if you want 节令月.
4. **大运 index 0** – First `DaYun` can have empty `getGanZhi()` (pre-birth or current); handle this in UI/logic.
5. **Solar term lookup** – For “which 节气 is this datetime?” use `getPrevJieQi()`/`getNextJieQi()` or iterate `getJieQiTable()`; boundaries are `Solar` with hour/minute/second (minute-accurate).

---

## 5. Test output reference (2003-04-24 21:20:00)

```
EightChar: 癸未 丙辰 丁卯 辛亥
Year: 癸未  WuXing: 水土  ShiShenGan: 七杀  ShiShenZhi: ['食神','比肩','偏印']  HideGan: ['己','丁','乙']
Month: 丙辰  WuXing: 火土  ShiShenGan: 劫财  ShiShenZhi: ['伤官','偏印','七杀']  HideGan: ['戊','乙','癸']
Day:   丁卯  WuXing: 火木  ShiShenGan: 日主  ShiShenZhi: ['偏印']  HideGan: ['乙']
Time:  辛亥  WuXing: 金水  ShiShenGan: 偏财  ShiShenZhi: ['正官','正印']  HideGan: ['壬','甲']

Yun(male): startYear=6, getDaYun()[0].getGanZhi()='', [1].getGanZhi()='乙卯', getStartYear()=2009
LiuNian[0]: year=2003, ganZhi=癸未
LiuYue[0]: ganZhi=甲寅, monthInChinese=正
PrevJieQi: 谷雨, Solar: 2003-04-20 20:02:48
```
