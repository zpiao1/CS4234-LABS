import kotlinx.coroutines.experimental.asCoroutineDispatcher
import kotlinx.coroutines.experimental.delay
import kotlinx.coroutines.experimental.launch
import kotlinx.coroutines.experimental.runBlocking
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit.SECONDS

private val MAXN = 120

class SimpleIteratedLocalSearch(
    val n: Int
) {

  private val T = Array(MAXN) { IntArray(MAXN) }
  val best = IntArray(MAXN)
  private val solution = IntArray(MAXN)
  private var T2 = Array(MAXN) { IntArray(MAXN) }
  private val C = IntArray(n)

  private var stayedFor = 0
  private var currentE = Integer.MAX_VALUE
  var bestE = Integer.MAX_VALUE

  private val random = Random()
  private val shuffleList = MutableList(n) { it }

  private suspend fun localSearch() {
    while (true) {
      var bestNeighborE = Integer.MAX_VALUE
      var bestIndex = -1
      for (idx in 0 until n) {
        val neighborE = valueFlip(idx)
        if (neighborE < bestNeighborE) {
          bestNeighborE = neighborE
          bestIndex = idx
        }
      }
      if (bestNeighborE >= currentE) {
        break
      }

      currentE = valueFlipAndUpdate(bestIndex)
      solution[bestIndex] *= -1
    }
  }

  private suspend fun generateInitialSolution() {
    for (i in 0 until n) {
      solution[i] = if (random.nextBoolean()) 1 else -1
    }
    stayedFor = 0
    recomputeDataStructure()
  }

  private suspend fun perturb() {
    val THRESHOLD = 10000
    if (stayedFor > THRESHOLD) {
      generateInitialSolution()
    } else {
      Collections.shuffle(shuffleList, random)
      for (i in 0 until 4) {
        val index = shuffleList[i]
        solution[index] *= -1
      }
      recomputeDataStructure()
    }
  }

  private suspend fun shouldAccept(): Boolean {
    if (currentE <= bestE) {
      if (currentE == bestE) {
        stayedFor++
      }
      bestE = currentE
      return true
    } else {
      return false
    }
  }

  private suspend fun saveBest() {
    bestE = currentE
    System.arraycopy(solution, 0, best, 0, n)
  }

  private suspend fun recomputeDataStructure() {
    currentE = 0
    for (k in 1 until n) {
      C[k] = 0
      for (i in 0 until n - k) {
        T[k][i] = solution[i] * solution[i + k]
        C[k] += T[k][i]
      }
      currentE += C[k] * C[k]
    }

    T2 = Array(MAXN) { IntArray(MAXN) }
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

  suspend fun run() {
    generateInitialSolution()
    localSearch()
    saveBest()
    while (true) {
      perturb()
      localSearch()
      if (shouldAccept()) {
        saveBest()
      }
    }
  }
}

fun main(args: Array<String>) {
  val timeoutSeconds = 30L
//  val timeoutHours = 1L
  val size = (110 - 66 + 1) * 20
  val array = Array(size) {
    val n = 66 + it / 20
    SimpleIteratedLocalSearch(n)
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
    delay(timeoutSeconds, SECONDS)
//    delay(timeoutHours, HOURS)
    parent.cancel()
    delay(1L, SECONDS)
//    delay(1L, MINUTES)
    array.groupBy { it.n }
        .map { (_, value) -> value.minBy { it.bestE }!! }
        .forEach {
          val solution = it.best
          println("n = ${it.n}")
          println("BF_OV: ${it.bestE}")
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