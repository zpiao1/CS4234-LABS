import kotlinx.coroutines.experimental.asCoroutineDispatcher
import kotlinx.coroutines.experimental.delay
import kotlinx.coroutines.experimental.launch
import kotlinx.coroutines.experimental.runBlocking
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit.HOURS
import java.util.concurrent.TimeUnit.MINUTES
import kotlin.system.measureTimeMillis

val BK_OV = intArrayOf(
    0, 0, 0, 1, 2, 2, 7, 3, 8, 12, 13,                 /* 0-10  */
    5, 10, 6, 19, 15, 24, 32, 25, 29, 26,              /* 11-20 */
    26, 39, 47, 36, 36, 45, 37, 50, 62, 59,            /* 21-30 */
    67, 64, 64, 65, 73, 82, 86, 87, 99, 108,           /* 31-40 */
    108, 101, 109, 122, 118, 131, 135, 140, 136, 153,  /* 41-50 */
    153, 166, 170, 175, 171, 192, 188, 197, 205, 218,  /* 51-60 */
    226, 235, 207, 208, 240, 265, 241, 250, 274, 295,  /* 61-70 (not yet proven to be optimal) */
    275, 300, 308, 349, 341, 338, 366                  /* 71-77 (not yet proven to be optimal) */
)
private val MAXN = 120

private fun BitSet.getValue(bitIndex: Int) = if (this[bitIndex]) 1 else -1

class TSv7Coroutine(
    val n: Int,
    private val MAX_STABLE: Int,
    private val PUTT: Int,
    private val THRESHOLD: Int,
    private val TABU_TENURE_LOW: Int,
    private val TABU_TENURE_DELTA: Int
) {

  private val solution = BitSet(MAXN)  // BitSet: true -> 1, false -> -1
  val bestSolution = BitSet(MAXN)
  private val refSolution = BitSet(MAXN)
  private val tabu = IntArray(MAXN)
  private val T = Array(MAXN, { IntArray(MAXN) })
  private var T2 = Array(MAXN, { IntArray(MAXN) })
  private val C = IntArray(MAXN)
  private var Cur_OV = 0
  var BF_OV = 0
  private val bestIdx = IntArray(MAXN)
  private var bestIdxID = 0
  private var bestNeighborOV = 0

  private val random = Random()
//      .apply { setSeed(1L) }

  private suspend fun valueFlip(i: Int): Int {
    var f = 0
    for (p in 1 until n) {
      val v = C[p] - T2[p][i]
      f += v * v
    }
    return f
  }

  private suspend fun valueFlipAndUpdate(i: Int): Int {
    var f = 0
    for (p in 1 until n) {
      var v = C[p]
      if (i + p < n) {      /* update right side */
        v -= 2 * T[p][i]
        T2[p][i] -= 4 * T[p][i]
        T2[p][i + p] -= 4 * T[p][i]
        T[p][i] *= -1    /* flip the value */
      }
      if (0 <= i - p) {     /* update left side */
        v -= 2 * T[p][i - p]
        T2[p][i] -= 4 * T[p][i - p]
        T2[p][i - p] -= 4 * T[p][i - p]
        T[p][i - p] *= -1 /* flip */
      }
      f += v * v
      C[p] = v           /* update C[p] */
    }
    return f
  }

  private suspend fun recomputeDataStructure() {
    Cur_OV = 0
    for (k in 1 until n) {
      C[k] = 0
      for (i in 0 until n - k) {
        T[k][i] = solution.getValue(i) * solution.getValue(i + k)
        C[k] += T[k][i]
      }
      Cur_OV += C[k] * C[k]
    }

    T2 = Array(n, { IntArray(n) })
    for (i in 0 until n) {
      var k = 1
      while (i - k >= 0 || n > i + k) {
        T2[k][i] = 0
        if (i + k < n) {
          T2[k][i] += 2 * T[k][i]
        }
        if (0 <= i - k) {
          T2[k][i] += 2 * T[k][i - k]
        }
        k++
      }
    }
  }

  private suspend fun localOptima() {
    while (true) {
      bestNeighborOV = Integer.MAX_VALUE
      bestIdxID = -1

      for (idx in 0 until n) {
        val neighborOV = valueFlip(idx)
        if (neighborOV < bestNeighborOV) {
          bestNeighborOV = neighborOV
          bestIdxID = 0
        }

        if (neighborOV == bestNeighborOV) {
          bestIdx[bestIdxID++] = idx
        }
      }

      if (bestNeighborOV >= Cur_OV) {
        break
      }

      val bI = bestIdx[random.nextInt(bestIdxID)]
      Cur_OV = valueFlipAndUpdate(bI)
      solution.flip(bI)
    }
  }

  private suspend fun randomConfiguration() {
    for (i in 0 until n) {
      solution[i] = random.nextBoolean()
    }
    recomputeDataStructure()
  }

  private suspend fun localRestartAndPerturbABit(PUTT: Int) {
    var d = 0
    for (i in 0 until n) {
      solution[i] = refSolution[i]
    }

    while (d < PUTT) {
      val randIndex = random.nextInt(n)
      if (solution[randIndex] == refSolution[randIndex]) {
        solution.flip(randIndex)
        d++
      }
    }

    recomputeDataStructure()
  }

  private suspend fun saveBest() {
    if (Cur_OV <= BF_OV) {
      BF_OV = Cur_OV
      for (i in 0 until n) {
        bestSolution[i] = solution[i]
        refSolution[i] = solution[i]
      }
    }
  }

  private suspend fun clearTabuTable() {
    Arrays.fill(tabu, Integer.MIN_VALUE)
  }

  suspend fun run() = measureTimeMillis {
    BF_OV = Integer.MAX_VALUE
    Cur_OV = Integer.MIN_VALUE
    clearTabuTable()
    randomConfiguration()
    saveBest()
    var k = 0
    var s = 0
    var localRestartCounter = 0

    while (true) {
      bestIdxID = 0
      bestNeighborOV = Integer.MAX_VALUE

      for (idx in 0 until n) {
        var OK = false
        if (k >= tabu[idx]) {
          Cur_OV = valueFlip(idx)
          OK = true
        }

        if (OK) {
          if (Cur_OV < bestNeighborOV) {
            bestNeighborOV = Cur_OV
            bestIdxID = 0
          }
          if (Cur_OV == bestNeighborOV) {
            bestIdx[bestIdxID++] = idx
          }
        }
      }

      val bI = bestIdx[random.nextInt(bestIdxID)]
      Cur_OV = valueFlipAndUpdate(bI)
      solution.flip(bI)

      val TABU_TENURE = TABU_TENURE_LOW + random.nextInt(TABU_TENURE_DELTA)

      tabu[bI] = k + TABU_TENURE

      if (Cur_OV <= BF_OV) {
        saveBest()
        if (Cur_OV == BF_OV) {
          s++
        } else {
          s = 0
          localRestartCounter = 0
        }
      } else if (s > MAX_STABLE) {
        localRestartCounter++
        if (localRestartCounter > THRESHOLD) {
          localRestartCounter = 0
          localRestartAndPerturbABit(PUTT / 2)
          localOptima()
          for (i in 0 until n) {
            refSolution[i] = solution[i]
          }
        } else {
          localRestartAndPerturbABit(PUTT)
        }
        clearTabuTable()
        s = 0
      } else {
        s++
      }
      k++

      saveBest()
    }
  }
}

fun main(args: Array<String>) {

  //  val timeoutSeconds = 30L
  val timeoutHours = 1L
  val size = (110 - 66 + 1) * 20
  val array = Array(size) {
    val n = 66 + it / 20
    val MAX_STABLE = 10 * n
    val PUTT = n / 4
    val THRESHOLD = 2 * n
    val TABU_TENURE_LOW = 1 + n / 20
    val TABU_TENURE_DELTA = n / 20
    TSv7Coroutine(n, MAX_STABLE, PUTT, THRESHOLD, TABU_TENURE_LOW, TABU_TENURE_DELTA)
  }
  val context = Executors.newWorkStealingPool(size).asCoroutineDispatcher()

  runBlocking {
    val parent = launch {
      val children = Array(size) {
        launch(context) { array[it].run() }
      }
      for (child in children) {
        child.join()
      }
    }
//    delay(timeoutSeconds, SECONDS)
    delay(timeoutHours, HOURS)
    parent.cancel()
    delay(1L, MINUTES)
    array.groupBy { it.n }
        .map { (_, value) -> value.minBy { it.BF_OV }!! }
        .forEach {
          val solution = it.bestSolution
          println("n = ${it.n}")
          println("BF_OV: ${it.BF_OV}")
          print("RLN: ")
          var count = 1
          for (i in 1 until it.n) {
            if (solution[i] != solution[i - 1]) {
              print(count)
              count = 1
            } else {
              count++
            }
          }
          println(count)  // print last digit
          print("Original bit: ")
          for (i in 0 until it.n) {
            print("${solution.getValue(i)} ")
          }
          println()
        }
  }
}