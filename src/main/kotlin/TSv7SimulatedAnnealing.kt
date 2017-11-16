import kotlinx.coroutines.experimental.asCoroutineDispatcher
import kotlinx.coroutines.experimental.delay
import kotlinx.coroutines.experimental.launch
import kotlinx.coroutines.experimental.runBlocking
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit.HOURS
import java.util.concurrent.TimeUnit.MINUTES

private val MAXN = 120

class TSv7SimulatedAnnealing(
    val n: Int,
    private val MAX_STABLE: Int,
    private val PUTT: Int,
    private val THRESHOLD: Int,
    private val TABU_TENURE_LOW: Int,
    private val TABU_TENURE_DELTA: Int,
    private var TEMP: Double,
    private val TEMP_FACTOR: Double
) {

  private val solution = IntArray(MAXN)
  val bestSolution = IntArray(MAXN)
  private val refSolution = IntArray(MAXN)
  private val tabu = IntArray(MAXN)
  private val T = Array(MAXN) { IntArray(MAXN) }
  private var T2 = Array(MAXN) { IntArray(MAXN) }
  private val C = IntArray(MAXN)
  private var Cur_OV = 0
  var BF_OV = 0
  private var bestIdx = IntArray(MAXN)
  private var bestIdxID = 0
  private var bestNeighborOV = 0
  private val neighbors = IntArray(MAXN)
  private val neighborIdx = IntArray(MAXN)
  private var neighborID = 0

  private val random = Random()

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
        T[k][i] = solution[i] * solution[i + k]
        C[k] += T[k][i]
      }
      Cur_OV += C[k] * C[k]
    }

    T2 = Array(n) { IntArray(n) }
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
      solution[bI] *= -1
    }
  }

  private suspend fun randomConfiguration() {
    for (i in 0 until n) {
      solution[i] = if (random.nextBoolean()) 1 else -1
    }
    recomputeDataStructure()
  }

  private suspend fun localRestartAndPerturbABit(PUTT: Int) {
    var d = 0

    System.arraycopy(refSolution, 0, solution, 0, n)

    while (d < PUTT) {
      val randIndex = random.nextInt(n)
      if (solution[randIndex] == refSolution[randIndex]) {
        solution[randIndex] *= -1
        d++
      }
    }

    recomputeDataStructure()
    TEMP = n.toDouble()
  }

  private suspend fun saveBest() {
    if (Cur_OV <= BF_OV) {
      BF_OV = Cur_OV
      System.arraycopy(solution, 0, bestSolution, 0, n)
      System.arraycopy(solution, 0, refSolution, 0, n)
    }
  }

  private suspend fun clearTabuTable() {
    Arrays.fill(tabu, Integer.MIN_VALUE)
  }

  suspend fun run() {
    Cur_OV = Integer.MAX_VALUE
    BF_OV = Integer.MAX_VALUE
    clearTabuTable()
    randomConfiguration()
    saveBest()
    var k = 0
    var s = 0
    var localRestartCounter = 0

    while (true) {
      bestIdxID = 0
      neighborID = 0
      bestNeighborOV = Integer.MAX_VALUE

      for (idx in 0 until n) {
        var OK = false

        if (k >= tabu[idx]) {
          Cur_OV = valueFlip(idx)
          OK = true
        }

        if (OK) {
          neighbors[neighborID] = Cur_OV
          neighborIdx[neighborID] = idx
          neighborID++
          if (Cur_OV < bestNeighborOV) {
            bestNeighborOV = Cur_OV
          }
        }
      }

      for (idx in 0 until neighborID) {
        if (neighbors[idx] == bestNeighborOV) {
          bestIdx[bestIdxID++] = neighborIdx[idx]
        } else {
          val acceptance = Math.exp((bestNeighborOV - neighbors[idx]) / TEMP)
          if (random.nextDouble() < acceptance) {
            bestIdx[bestIdxID++] = neighborIdx[idx]
          }
        }
      }

      val bI = bestIdx[random.nextInt(bestIdxID)]
      Cur_OV = valueFlipAndUpdate(bI)
      solution[bI] *= -1

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
          System.arraycopy(solution, 0, refSolution, 0, n)
        } else {
          localRestartAndPerturbABit(PUTT)
        }
        clearTabuTable()
        s = 0
      } else {
        s++
      }
      k++
      TEMP *= TEMP_FACTOR

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
    val TEMP = n.toDouble()
    val TEMP_FACTOR = 0.98
    TSv7SimulatedAnnealing(
        n,
        MAX_STABLE,
        PUTT,
        THRESHOLD,
        TABU_TENURE_LOW,
        TABU_TENURE_DELTA,
        TEMP,
        TEMP_FACTOR
    )
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
//    delay(1L, SECONDS)
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
            print("${solution[i]} ")
          }
          println()
        }
  }
}