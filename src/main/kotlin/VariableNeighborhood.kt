import kotlinx.coroutines.experimental.asCoroutineDispatcher
import kotlinx.coroutines.experimental.delay
import kotlinx.coroutines.experimental.launch
import kotlinx.coroutines.experimental.runBlocking
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit.HOURS
import java.util.concurrent.TimeUnit.MINUTES

private val MAXN = 120

class VariableNeighborhood(
    val n: Int,
    val MAX_STABLE: Int,
    val THRESHOLD: Int,
    val PUTT: Int
) {

  private val solution = IntArray(MAXN)
  val bestSolution = IntArray(MAXN)
  private val refSolution = IntArray(MAXN)
  private val T = Array(MAXN) { IntArray(MAXN) }
  private var T2 = Array(MAXN) { IntArray(MAXN) }
  private val C = IntArray(MAXN)
  private var Cur_OV = 0
  var BF_OV = 0
  private val bestIdx = IntArray(MAXN)

  private val shuffleList = MutableList(n) { it }

  private val random = Random()

  private suspend fun flipKValues(k: Int) {
    Collections.shuffle(shuffleList, random)
    for (i in 0 until k) {
      valueFlipAndUpdate(shuffleList[i])
      solution[shuffleList[i]] *= -1
    }
    recomputeDataStructure()
  }

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
      var bestNeighborOV = Integer.MAX_VALUE
      var bestIdxID = -1

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
    System.arraycopy(solution, 0, refSolution, 0, n)
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
  }

  private suspend fun saveBest() {
    if (Cur_OV <= BF_OV) {
      BF_OV = Cur_OV
      System.arraycopy(solution, 0, bestSolution, 0, n)
      System.arraycopy(solution, 0, refSolution, 0, n)
    }
  }

  suspend fun run() {
    Cur_OV = Integer.MAX_VALUE
    BF_OV = Integer.MAX_VALUE
    randomConfiguration()
    saveBest()
    var s = 0
    var randomRestart = 0
    while (true) {
      var k = 1
      while (k < n) {
        if (k < n / 4) {
          flipKValues(k)
        } else {
          localRestartAndPerturbABit(k)
        }
        localOptima()
        if (Cur_OV < BF_OV) {
          saveBest()
          k = 1
          s = 0
          randomRestart = 0
        } else {
          s++
          k++
        }
        saveBest()

        if (s > MAX_STABLE) {
          randomRestart++
          if (randomRestart > THRESHOLD) {
            randomRestart = 0
            randomConfiguration()
          } else {
            localRestartAndPerturbABit(PUTT / 2)
          }
          localOptima()
          System.arraycopy(solution, 0, refSolution, 0, n)
          k = 1
          s = 0
        }
      }
    }
  }
}

fun main(args: Array<String>) {
//  val timeoutSeconds = 30L
  val timeoutHours = 1L
  val size = (110 - 66 + 1) * 20
  val array = Array(size) {
    val n = 66 + it / 20
    val MAX_STABLE = 2 * n
    val PUTT = n / 4
    val THRESHOLD = 10
    VariableNeighborhood(n, MAX_STABLE, THRESHOLD, PUTT)
  }
  val context = Executors.newWorkStealingPool(size)
      .asCoroutineDispatcher()

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
              print(if (count < 10) count else 'A' + (count - 10))
              count = 1
            } else {
              count++
            }
          }
          println(if (count < 10) count else 'A' + (count - 10))  // print last digit
          print("Original bit: ")
          for (i in 0 until it.n) {
            print("${solution[i]} ")
          }
          println()
        }
  }
}