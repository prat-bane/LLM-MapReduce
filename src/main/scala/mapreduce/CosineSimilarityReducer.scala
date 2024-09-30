package mapreduce

import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.Reducer
import org.apache.log4j.Logger
import scala.collection.mutable
import scala.io.Source

class CosineSimilarityReducer extends Reducer[Text, Text, Text, Text] {

  private val logger = Logger.getLogger(classOf[CosineSimilarityReducer])
  private val topN = 5 // Number of top similar words to keep

  // Map to store all embeddings loaded from Distributed Cache
  private val allEmbeddings = mutable.Map[String, Array[Double]]()

  override def setup(context: Reducer[Text, Text, Text, Text]#Context): Unit = {
    // Load all embeddings from the Distributed Cache
    val cacheFiles = context.getCacheFiles
    if (cacheFiles != null && cacheFiles.length > 0) {
      try {
        val cachePath = cacheFiles(0).toString
        logger.info(s"Loading embeddings from Distributed Cache: $cachePath")
        val source = Source.fromURL(cacheFiles(0).toURL)
        for (line <- source.getLines()) {
          val parts = line.trim.split("\t")
          if (parts.length == 2) {
            val word = parts(0).trim
            val embeddingStr = parts(1).trim
            val embedding = embeddingStr.split(",").map(_.toDouble)
            if (embedding.nonEmpty) {
              allEmbeddings += (word -> embedding)
            }
          }
        }
        source.close()
        logger.info(s"Loaded ${allEmbeddings.size} embeddings from Distributed Cache.")
      } catch {
        case e: Exception =>
          logger.error("Error loading embeddings from Distributed Cache", e)
          throw new RuntimeException(e)
      }
    } else {
      logger.error("No embeddings file found in Distributed Cache.")
      throw new RuntimeException("Embeddings file not found in Distributed Cache.")
    }
  }

  override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
    val word = key.toString
    val iterator = values.iterator()
    if (iterator.hasNext) {
      val embeddingStr = iterator.next().toString.trim
      val embedding = embeddingStr.split(",").map(_.toDouble)
      if (embedding.nonEmpty) {
        // Compute norm of the current word's embedding
        val norm1 = math.sqrt(embedding.map(x => x * x).sum)

        // Mutable priority queue to keep top N similar words (min-heap)
        implicit val ord: Ordering[(String, Double)] = Ordering.by(-_._2) // Max-heap
        val topSimilarities = mutable.PriorityQueue.empty[(String, Double)]

        // Iterate over all embeddings to compute cosine similarity
        allEmbeddings.foreach { case (otherWord, otherEmbedding) =>
          if (otherWord != word) {
            // Compute norm of the other word's embedding
            val norm2 = math.sqrt(otherEmbedding.map(x => x * x).sum)

            // Compute dot product
            val dotProduct = (embedding zip otherEmbedding).map { case (x, y) => x * y }.sum

            // Compute cosine similarity
            val similarity = if (norm1 != 0.0 && norm2 != 0.0) dotProduct / (norm1 * norm2) else 0.0

            // Add to priority queue
            topSimilarities.enqueue((otherWord, similarity))

            // Keep only top N
            if (topSimilarities.size > topN) {
              topSimilarities.dequeue()
            }
          }
        }

        // Extract top N similar words and sort them in descending order
        val topSimilar = topSimilarities.toList.sortBy(-_._2)

        // Format the output: similarword1:cosineSimilarity similarword2:cosineSimilarity ...
        val topSimilarStr = topSimilar.map { case (w, s) => s"$w:$s" }.mkString(" ")

        // Emit the result
        context.write(new Text(word), new Text(topSimilarStr))
      }
    }
  }

  override def cleanup(context: Reducer[Text, Text, Text, Text]#Context): Unit = {
    logger.info("Completed processing all assigned words for cosine similarity.")
  }
}
