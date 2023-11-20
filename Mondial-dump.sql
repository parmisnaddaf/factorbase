-- MySQL dump 10.13  Distrib 5.5.34, for Linux (x86_64)
--
-- Host: localhost    Database: Mondial
-- ------------------------------------------------------
-- Server version	5.5.34

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Current Database: `Mondial`
--

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `Mondial` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `Mondial`;

--
-- Table structure for table `border`
--

DROP TABLE IF EXISTS `border`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `border` (
  `c1_id` int(10) unsigned NOT NULL DEFAULT '0',
  `c1_id_dummy` int(10) unsigned NOT NULL DEFAULT '0',
  PRIMARY KEY (`c1_id`,`c1_id_dummy`),
  KEY `FK_border_2` (`c1_id`),
  KEY `FK_border_1` (`c1_id_dummy`),
  CONSTRAINT `FK_border_1` FOREIGN KEY (`c1_id_dummy`) REFERENCES `country` (`c1_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `FK_border_2` FOREIGN KEY (`c1_id`) REFERENCES `country` (`c1_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 ROW_FORMAT=FIXED;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `border`
--

LOCK TABLES `border` WRITE;
/*!40000 ALTER TABLE `border` DISABLE KEYS */;
INSERT INTO `border` VALUES (1,35),(1,45),(1,46),(1,63),(1,78),(1,82),(1,167),(1,169),(3,73),(3,110),(3,111),(5,51),(5,59),(6,119),(6,144),(6,207),(6,208),(7,11),(7,69),(7,85),(7,186),(11,7),(11,69),(11,85),(11,186),(12,46),(12,59),(12,96),(12,125),(13,84),(15,154),(15,161),(19,73),(19,110),(19,155),(19,174),(19,186),(20,84),(20,181),(21,162),(21,208),(22,81),(22,174),(23,24),(23,131),(23,137),(23,139),(23,145),(24,23),(24,38),(24,61),(24,77),(24,131),(24,137),(24,139),(24,157),(24,170),(24,206),(29,134),(30,108),(32,66),(32,72),(32,142),(32,144),(32,179),(34,191),(35,1),(35,46),(35,59),(35,63),(35,82),(36,70),(36,99),(36,147),(36,153),(38,24),(38,55),(38,130),(38,131),(38,206),(42,123),(42,130),(45,1),(45,46),(45,134),(45,167),(46,1),(46,12),(46,35),(46,45),(46,48),(46,59),(46,96),(46,125),(46,134),(47,58),(48,46),(49,148),(50,102),(50,150),(50,153),(50,154),(50,184),(50,204),(51,5),(51,59),(51,68),(51,129),(52,53),(52,58),(53,52),(53,162),(53,208),(55,38),(55,131),(56,79),(57,83),(58,47),(58,52),(59,5),(59,12),(59,35),(59,46),(59,51),(59,82),(59,96),(59,106),(61,24),(61,170),(63,1),(63,35),(66,32),(66,72),(66,144),(68,51),(69,7),(69,11),(69,186),(70,36),(70,161),(71,147),(71,171),(72,32),(72,66),(73,3),(73,19),(73,110),(73,186),(77,24),(77,170),(77,206),(78,1),(78,81),(78,155),(78,167),(78,169),(78,174),(79,56),(79,123),(81,22),(81,78),(81,111),(81,169),(81,174),(82,1),(82,35),(82,59),(82,160),(82,169),(82,193),(83,57),(83,91),(83,151),(83,177),(84,13),(84,20),(84,122),(84,133),(84,181),(85,7),(85,11),(85,87),(85,133),(85,183),(85,186),(87,85),(87,91),(87,95),(87,164),(87,177),(87,186),(91,83),(91,87),(91,164),(91,177),(92,97),(92,180),(93,181),(93,183),(93,192),(95,87),(95,164),(96,12),(96,46),(96,59),(97,92),(97,180),(97,181),(99,36),(99,147),(99,198),(100,159),(102,50),(102,204),(106,59),(107,155),(108,30),(108,191),(110,3),(110,19),(110,73),(110,174),(111,3),(111,81),(111,174),(112,181),(113,116),(113,159),(113,165),(113,207),(113,209),(116,113),(116,207),(118,163),(118,166),(119,6),(119,140),(119,159),(119,207),(122,84),(122,181),(123,42),(123,79),(125,12),(125,46),(128,164),(129,51),(130,38),(130,42),(131,23),(131,24),(131,38),(131,55),(131,145),(133,84),(133,85),(133,181),(134,29),(134,45),(134,46),(134,167),(135,149),(137,23),(137,24),(137,139),(138,164),(139,23),(139,24),(139,137),(139,145),(139,157),(140,119),(140,159),(140,209),(142,32),(142,144),(142,179),(142,208),(144,6),(144,32),(144,66),(144,142),(144,208),(145,23),(145,131),(145,139),(147,36),(147,71),(147,99),(147,153),(147,171),(147,198),(148,49),(149,135),(149,182),(150,50),(150,153),(150,171),(150,204),(151,83),(151,177),(153,36),(153,50),(153,147),(153,150),(153,154),(153,171),(154,15),(154,50),(154,153),(154,179),(155,19),(155,78),(155,107),(155,174),(157,24),(157,139),(159,100),(159,113),(159,119),(159,140),(159,165),(159,209),(160,82),(161,15),(161,70),(162,21),(162,53),(162,208),(163,118),(163,166),(164,87),(164,91),(164,95),(164,128),(164,138),(165,113),(165,159),(166,118),(166,163),(167,1),(167,45),(167,78),(167,134),(169,1),(169,78),(169,81),(169,82),(170,24),(170,61),(170,77),(171,71),(171,147),(171,150),(171,153),(171,197),(174,19),(174,22),(174,78),(174,81),(174,110),(174,111),(174,155),(177,83),(177,87),(177,91),(177,151),(177,186),(178,181),(178,192),(179,32),(179,142),(179,154),(180,92),(180,97),(181,20),(181,84),(181,93),(181,97),(181,112),(181,122),(181,133),(181,178),(182,149),(183,85),(183,93),(183,192),(184,50),(186,7),(186,11),(186,19),(186,69),(186,73),(186,85),(186,87),(186,177),(191,34),(191,108),(192,93),(192,178),(192,183),(193,82),(197,171),(198,99),(198,147),(204,50),(204,102),(204,150),(206,24),(206,38),(206,77),(207,6),(207,113),(207,116),(207,119),(207,208),(207,209),(208,6),(208,21),(208,53),(208,142),(208,144),(208,162),(208,207),(209,113),(209,140),(209,159),(209,207);
/*!40000 ALTER TABLE `border` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `country`
--

DROP TABLE IF EXISTS `country`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `country` (
  `c1_id` int(10) unsigned NOT NULL DEFAULT '0',
  `govern` varchar(120) NOT NULL DEFAULT '0',
  `continent` varchar(20) NOT NULL,
  `percentage` float DEFAULT NULL,
  `popu` int(11) DEFAULT NULL,
  `class` int(10) unsigned NOT NULL DEFAULT '0',
  PRIMARY KEY (`c1_id`),
  KEY `country_govern` (`govern`),
  KEY `country_continent` (`continent`),
  KEY `country_percentage` (`percentage`),
  KEY `country_popu` (`popu`),
  KEY `country_class` (`class`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 ROW_FORMAT=FIXED;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `country`
--

LOCK TABLES `country` WRITE;
/*!40000 ALTER TABLE `country` DISABLE KEYS */;
INSERT INTO `country` VALUES (1,'republic','Europe',2,3,1),(3,'democ','Europe',1,2,0),(4,'dependent','aus',1,0,1),(5,'democ','Europe',4,0,1),(6,'democ','Africa',0,3,1),(7,'republic','Asia',3,2,1),(8,'dependent','America',2,0,1),(10,'dependent','America',0,0,1),(11,'republic','Asia',3,3,0),(12,'monarch','Europe',2,3,1),(13,'republic','Asia',2,4,0),(14,'democ','America',1,0,1),(15,'democ','Africa',0,2,0),(17,'dependent','America',0,0,0),(19,'democ','Europe',2,3,1),(20,'monarch','Asia',2,1,0),(21,'republic','Africa',1,2,1),(22,'democ','Europe',0,2,0),(23,'republic','America',3,3,1),(24,'republic','America',1,4,1),(25,'monarch','Asia',4,1,0),(28,'dependent','America',2,0,1),(29,'republic','Europe',1,3,1),(30,'democ','America',1,0,1),(31,'communist','America',2,3,1),(32,'republic','Africa',0,3,1),(33,'dependent','America',0,0,0),(34,'democ','America',0,4,1),(35,'republic','Europe',0,3,1),(36,'republic','Africa',1,3,0),(37,'republic','Asia',1,3,0),(38,'republic','America',3,4,1),(39,'dependent','aus',2,0,0),(40,'republic','Africa',2,1,0),(42,'democ','America',3,2,1),(43,'republic','Africa',4,1,1),(44,'republic','Europe',2,1,1),(45,'democ','Europe',0,3,1),(46,'republic','Europe',0,4,1),(47,'republic','Africa',3,1,0),(48,'monarch','Europe',3,2,1),(49,'republic','America',3,3,1),(50,'republic','Africa',3,4,0),(51,'monarch','Europe',3,4,1),(52,'republic','Africa',0,4,1),(53,'republic','Africa',0,4,1),(55,'republic','America',3,3,1),(56,'republic','America',2,2,1),(57,'republic','Asia',3,4,0),(58,'republic','Africa',0,4,0),(59,'republic','Europe',3,4,1),(60,'dependent','Europe',4,0,1),(61,'dependent','America',4,0,1),(62,'republic','aus',1,1,1),(63,'monarch','Europe',2,0,1),(64,'dependent','aus',1,1,1),(66,'republic','Africa',1,1,1),(68,'dependent','Europe',1,0,1),(69,'republic','Asia',2,2,1),(70,'democ','Africa',0,3,0),(71,'republic','Africa',0,1,0),(72,'democ','Africa',4,1,1),(73,'republic','Europe',3,3,1),(74,'dependent','America',4,0,1),(75,'dependent','America',3,1,1),(77,'republic','America',1,1,1),(78,'republic','Europe',1,3,1),(79,'republic','America',3,2,1),(81,'democ','Europe',2,2,1),(82,'republic','Europe',3,4,1),(83,'republic','Asia',2,2,0),(84,'republic','Asia',2,4,0),(85,'republic','Asia',3,4,0),(86,'republic','Europe',3,2,1),(87,'republic','Asia',3,4,0),(88,'republic','Europe',3,1,1),(89,'monarch','Asia',2,4,0),(90,'democ','America',1,2,1),(91,'monarch','Asia',3,2,0),(92,'democ','Asia',3,3,0),(93,'republic','Asia',0,3,0),(94,'republic','aus',1,0,1),(95,'monarch','Asia',2,1,0),(96,'monarch','Europe',3,1,1),(97,'communist','Asia',1,2,0),(99,'republic','Africa',0,1,0),(100,'monarch','Africa',2,1,1),(101,'democ','Europe',3,1,1),(102,'monarch','Africa',3,4,0),(104,'dependent','America',3,1,1),(105,'dependent','Africa',3,0,0),(106,'monarch','Europe',3,0,1),(107,'republic','Europe',3,2,1),(108,'republic','America',2,4,1),(110,'democ','Europe',1,1,1),(111,'democ','Europe',2,1,1),(112,'republic','Asia',0,2,0),(113,'republic','Africa',0,3,1),(114,'democ','Africa',1,1,0),(115,'republic','Asia',4,1,0),(116,'democ','Africa',1,3,1),(118,'monarch','Europe',2,2,1),(119,'republic','Africa',1,1,1),(120,'republic','aus',4,0,1),(121,'dependent','aus',1,0,1),(122,'democ','Asia',3,4,0),(123,'republic','America',3,2,1),(124,'dependent','aus',1,0,0),(125,'monarch','Europe',0,3,1),(126,'dependent','aus',0,0,0),(127,'democ','aus',0,2,0),(128,'monarch','Asia',2,1,0),(129,'republic','Europe',3,3,1),(130,'republic','America',2,2,1),(131,'republic','America',4,4,1),(132,'dependent','aus',4,0,0),(133,'republic','Asia',3,4,0),(134,'democ','Europe',3,4,1),(135,'democ','aus',0,2,1),(137,'republic','America',3,2,1),(138,'monarch','Asia',3,1,0),(139,'republic','America',3,4,1),(140,'republic','Africa',1,1,1),(141,'democ','Asia',3,4,0),(142,'republic','Africa',0,2,1),(144,'republic','Africa',1,2,1),(145,'republic','America',2,3,1),(146,'dependent','aus',3,1,1),(147,'republic','Africa',2,3,0),(148,'republic','America',2,2,1),(149,'republic','Asia',2,4,0),(150,'republic','Africa',4,1,0),(151,'republic','Asia',1,2,0),(152,'republic','Africa',0,3,1),(153,'republic','Africa',3,3,0),(154,'republic','Africa',2,3,0),(155,'republic','Europe',1,4,1),(156,'republic','Asia',0,4,1),(157,'republic','America',1,2,1),(158,'republic','Asia',2,4,1),(159,'republic','Africa',0,4,0),(160,'republic','Europe',4,0,1),(161,'democ','Africa',0,2,1),(162,'republic','Africa',1,2,1),(163,'monarch','Europe',3,3,1),(164,'monarch','Asia',4,3,0),(165,'monarch','Africa',1,1,1),(166,'republic','Europe',2,2,1),(167,'democ','Europe',1,2,1),(168,'democ','aus',0,1,0),(169,'democ','Europe',3,1,1),(170,'republic','America',0,1,0),(171,'democ','Africa',3,3,0),(173,'dependent','America',3,0,1),(174,'democ','Europe',2,3,1),(176,'republic','Africa',3,0,1),(177,'republic','Asia',3,3,0),(178,'republic','Asia',2,2,0),(179,'republic','Africa',1,2,0),(180,'monarch','Asia',3,4,0),(181,'communist','Asia',0,4,0),(182,'democ','Asia',3,1,1),(183,'republic','Asia',2,2,0),(184,'republic','Africa',3,3,0),(185,'monarch','aus',4,0,1),(186,'democ','Asia',3,4,0),(187,'democ','America',0,1,1),(188,'dependent','America',0,0,0),(189,'democ','aus',3,0,0),(191,'republic','America',1,4,1),(192,'republic','Asia',2,4,0),(193,'monarch','Europe',4,0,1),(195,'republic','aus',0,0,0),(196,'dependent','aus',3,0,1),(197,'democ','Africa',3,1,0),(198,'democ','Africa',1,2,0),(200,'democ','America',2,0,1),(202,'democ','America',3,0,1),(203,'monarch','aus',3,0,1),(204,'dependent','Africa',4,0,0),(205,'dependent','aus',0,0,0),(206,'republic','America',3,4,1),(207,'republic','Africa',1,3,1),(208,'republic','Africa',1,4,1),(209,'democ','Africa',0,3,1);
/*!40000 ALTER TABLE `country` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `eco`
--

DROP TABLE IF EXISTS `eco`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `eco` (
  `gdp` float DEFAULT NULL,
  `agricul` float DEFAULT NULL,
  `service` float DEFAULT NULL,
  `industry` float DEFAULT NULL,
  `inflation` float DEFAULT NULL,
  `eco_id` int(10) unsigned NOT NULL DEFAULT '0',
  PRIMARY KEY (`eco_id`),
  KEY `eco_gdp` (`gdp`),
  KEY `eco_agricul` (`agricul`),
  KEY `eco_service` (`service`),
  KEY `eco_industry` (`industry`),
  KEY `eco_inflation` (`inflation`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `eco`
--

LOCK TABLES `eco` WRITE;
/*!40000 ALTER TABLE `eco` DISABLE KEYS */;
INSERT INTO `eco` VALUES (3,0,3,3,0,1),(1,2,4,1,0,2),(1,4,3,0,0,3),(4,0,2,3,0,4),(4,0,2,3,0,5),(0,1,4,2,0,6),(1,4,0,2,1,7),(1,0,0,4,0,8),(1,3,1,2,0,9),(2,2,3,2,0,10),(0,4,1,1,0,11),(4,2,2,2,0,12),(1,0,4,2,0,13),(1,0,3,3,0,14),(3,2,4,1,4,15),(2,3,1,2,0,16),(0,0,0,4,0,17),(4,0,2,3,0,18),(3,0,3,3,0,19),(2,4,1,1,0,20),(3,3,1,2,0,21),(4,2,2,2,0,22),(0,2,0,3,0,23),(0,2,1,3,0,24),(3,1,4,2,0,25),(0,0,1,3,0,26),(3,0,1,3,0,27),(2,2,3,2,0,28),(3,2,4,1,0,29),(4,1,3,3,0,30),(2,3,1,2,0,31),(2,4,0,1,0,32),(3,2,4,2,0,33),(2,4,0,1,0,34),(4,0,2,3,0,35),(1,3,1,2,0,36),(1,0,1,3,0,37),(1,1,4,2,0,38),(1,4,0,0,0,39),(2,4,0,1,1,40),(0,4,0,2,0,41),(0,4,2,1,0,42),(3,2,1,3,0,43),(0,1,0,4,0,44),(0,3,2,2,0,45),(3,1,3,2,0,46),(2,3,1,2,0,47),(4,0,0,4,0,48),(2,2,2,2,0,49),(4,0,3,3,0,50),(3,0,1,3,0,51),(4,2,3,1,1,52),(3,1,3,2,0,53),(1,1,1,3,0,54),(4,0,4,2,0,55),(1,1,2,3,0,56),(2,2,2,3,0,57),(1,4,0,1,0,58),(3,3,4,1,1,59),(2,0,4,2,0,60),(1,0,3,3,0,61),(1,4,1,1,0,62),(0,1,4,1,0,63),(3,2,3,2,0,64),(4,0,0,4,0,65),(0,1,0,4,0,66),(1,3,3,1,0,67),(4,1,2,3,1,68),(0,3,4,1,0,69),(1,3,3,1,1,70),(2,3,0,2,1,71),(0,2,0,3,0,72),(1,3,0,2,1,73),(3,4,0,1,0,74),(3,0,3,3,0,75),(0,2,0,3,0,76),(2,4,1,1,0,77),(0,3,2,2,0,78),(4,0,2,3,0,79),(3,1,2,3,0,80),(2,0,4,1,0,81),(3,1,3,2,0,82),(2,1,0,3,0,83),(4,3,2,2,0,84),(4,1,4,2,0,85),(3,0,4,2,0,86),(2,3,2,2,0,87),(1,0,4,2,0,88),(4,1,3,3,0,89),(4,1,3,2,0,90),(0,4,0,1,0,91),(1,2,3,2,1,92),(3,1,3,2,0,93),(1,3,3,2,0,94),(1,4,1,1,0,95),(4,2,3,2,0,96),(0,3,2,2,0,97),(2,2,2,2,0,98),(2,4,0,2,0,99),(1,4,0,1,0,100),(1,4,1,2,0,101),(3,2,3,2,0,102),(4,1,4,2,0,103),(2,2,2,3,0,104),(4,3,2,2,0,105),(0,4,1,1,0,106),(0,4,0,1,1,107),(3,0,2,3,0,108),(4,1,4,1,0,109),(0,3,4,1,0,110),(3,1,2,3,0,111),(2,1,4,2,0,112),(2,1,4,2,0,113),(0,3,1,2,1,114),(3,2,2,3,0,115),(2,3,1,2,1,116),(3,3,2,2,0,117),(0,4,1,1,0,118),(4,1,2,2,0,119),(4,2,4,1,0,120),(0,3,0,2,0,121),(2,3,3,1,0,122),(2,2,2,2,0,123),(4,2,3,2,1,124),(2,1,4,2,0,125),(3,0,4,1,0,126),(4,0,1,3,0,127),(0,3,0,2,0,128),(1,4,1,1,0,129),(3,4,1,1,1,130),(0,3,0,2,0,131),(0,2,1,3,0,132),(4,1,4,2,1,133),(1,3,1,2,1,134),(2,2,3,2,0,135);
/*!40000 ALTER TABLE `eco` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ecoR`
--

DROP TABLE IF EXISTS `ecoR`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ecoR` (
  `c1_id` int(10) unsigned NOT NULL DEFAULT '0',
  `eco_id` int(10) unsigned NOT NULL DEFAULT '0',
  PRIMARY KEY (`c1_id`,`eco_id`),
  KEY `FK_ecoR_1` (`c1_id`),
  KEY `FK_ecoR_2` (`eco_id`),
  CONSTRAINT `FK_ecoR_1` FOREIGN KEY (`c1_id`) REFERENCES `country` (`c1_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `FK_ecoR_2` FOREIGN KEY (`eco_id`) REFERENCES `eco` (`eco_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 ROW_FORMAT=FIXED;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ecoR`
--

LOCK TABLES `ecoR` WRITE;
/*!40000 ALTER TABLE `ecoR` DISABLE KEYS */;
INSERT INTO `ecoR` VALUES (1,1),(6,2),(7,3),(12,5),(14,6),(15,7),(17,8),(19,10),(21,11),(24,12),(29,15),(32,16),(33,17),(34,18),(35,19),(36,20),(37,21),(38,22),(43,24),(45,25),(47,26),(48,27),(49,28),(50,29),(51,30),(52,31),(53,32),(55,33),(58,34),(59,35),(62,36),(64,37),(66,38),(69,39),(70,40),(71,41),(72,42),(73,43),(75,44),(77,45),(78,46),(79,47),(81,49),(82,50),(83,51),(85,52),(86,53),(88,54),(89,55),(90,56),(91,57),(92,58),(93,59),(95,60),(96,61),(97,62),(100,63),(102,64),(104,66),(107,67),(108,68),(110,69),(112,70),(113,71),(115,72),(116,73),(118,75),(121,76),(122,77),(124,78),(125,79),(127,80),(128,81),(129,82),(130,83),(133,84),(134,85),(137,87),(138,88),(139,89),(141,90),(142,91),(144,92),(145,93),(147,94),(148,95),(149,96),(150,97),(151,98),(152,99),(153,100),(154,101),(155,102),(156,103),(157,104),(158,105),(161,106),(162,107),(163,108),(164,109),(165,110),(166,111),(167,112),(169,113),(170,114),(174,115),(177,117),(179,118),(180,119),(181,120),(182,121),(183,122),(184,123),(186,124),(187,125),(191,127),(197,128),(198,129),(202,132),(206,133),(207,134),(209,135);
/*!40000 ALTER TABLE `ecoR` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-05-10 15:24:15
