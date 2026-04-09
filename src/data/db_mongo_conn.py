from urllib.parse import quote_plus
import pymongo
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import datetime
import subprocess

import dotenv

log = logging.getLogger('kpub')


def from_env(database, collection=None):
    """Create a MongoDBConnector using connection details from .env."""
    config = {
        database: {
            "server": dotenv.get_key(".env", "MONGO_SERVER"),
            "port": int(dotenv.get_key(".env", "MONGO_PORT") or 27017),
            "user": dotenv.get_key(".env", "MONGO_USER") or "",
            "pwd": dotenv.get_key(".env", "MONGO_PWD") or "",
            "collection": collection or "articles",
        }
    }
    return MongoDBConnector(config, database, collection)


class MongoDBConnector:

    def __init__(self, config, database, collection=None):

        self.error = None
        self.readonly = False

        # parse config file
        if database not in config.keys():
            self.error = "DATABASE_CONFIG_ERROR"
            return None
        self.dbconfig = config[database]

        self.client = None
        collection = collection if collection else self.dbconfig["collection"]
        self.connect(database, collection)

    def connect(self, database, collection):
        """
        Connect to the specified database.  If primary server is down and backup
        is specified, then connect to it.  This also set the readOnly flag to 1.
        """

        # get db connect data
        server = self.dbconfig["server"] + ":" + str(self.dbconfig["port"])
        readonlyserver = self.dbconfig.get("readonlyserver", server)
        cmd = ["ping", "-c", "1", "-W", "1", self.dbconfig["server"]]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.wait(timeout=2)
            if p.returncode != 0:
                server = readonlyserver
                self.readonly = True
        except Exception:
            server = readonlyserver
            self.readonly = True

        user = self.dbconfig["user"]
        pwd = self.dbconfig["pwd"]
        try:
            if user and pwd:
                server = f"{user}:{quote_plus(pwd)}@{quote_plus(server)}"
            url = f"mongodb://{server}/{database}?authSource=admin"
            self.client = MongoClient(url)
        except ConnectionFailure:
            self.error = "PYMONGO_CONNECTION_ERROR"
            self.client = None

        self.collection = self.client[database][collection]

    def __del__(self):
        """Destructor to close the MongoDB connection."""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except Exception as e:
                log.error(f"Error closing MongoDB connection: {e}")
            finally:
                self.client = None

    def add_row(self, article, month, year, mission, 
                snippits, instruments, archive, affiliation, 
                reason, hasAcknowledgement=False):
        """Insert a document into the MongoDB collection."""
        try:
            # Use bibcode as the unique identifier
            article['_id'] = article['bibcode']

            article['last_modifier'] = 'kpub'
            article['date_modified'] = datetime.datetime.now()
            article['month'] = int(month)
            article['year'] = int(year)
            article['mission'] = mission
            article['snippits'] = snippits
            article['instruments'] = instruments.split('|')  # Convert to array
            article['archive'] = archive
            article['affiliation'] = affiliation
            article['reason'] = reason
            article['has_acknowledgement'] = hasAcknowledgement
            self.collection.insert_one(article)
            #self.collection.replace_one({'_id': article['_id']}, article, upsert=True)
            log.info(f"Inserted {article['bibcode']}")
        except pymongo.errors.DuplicateKeyError:
            log.warning(f"{article['bibcode']} was already ingested.")

    def update_row_affiliation(self, article):
        """Update a document in the MongoDB collection."""
        try:
            self.collection.update_one({'_id': article['_id']}, {'$set': {
                'last_modifier': article['last_modifier'],
                'date_modified': article['date_modified'],
                'affiliation': article['affiliation']
            }})
            log.info(f"Updated {article['bibcode']}")
            return article
        except Exception as e:
            log.error(f"Error updating {article['bibcode']}: {e}")

    def delete_by_bibcode(self, bibcode):
        """Delete a document by bibcode."""
        result = self.collection.delete_one({'bibcode': bibcode})
        log.info(f"Deleted {result.deleted_count} document(s).")

    def query(self, mission=None, year=None):
        """Query the MongoDB collection."""
        query = {}
        if mission:
            query['mission'] = mission
        else:
            query['mission'] = {'$ne': 'unrelated'}

        if year:
            if isinstance(year, (list, tuple)):
                query['year'] = {'$in': year}
            else:
                query['year'] = year

        rows = list(self.collection.find(
            query).sort('date', pymongo.DESCENDING))
        return rows

    def get_metrics_data(self, year_begin, year_end):

        match = {'$match': 
                   { 'year': {'$gte': year_begin, '$lte': year_end}, 
                    'affiliation': 'keck' }
                }
        unwind = {'$unwind': '$author_norm'}
        group = {'$group': 
                  {'_id': '$year', 
                   'author_count': {'$sum': 1}, 
                   'author_set': {'$addToSet': '$author_norm'}, 
                   'first_author_set': {'$addToSet': '$first_author_norm'}, 
                   'bibcodes': {'$addToSet': '$bibcode'}}}
        project = {'$project': 
                    {'paper_count': {'$size': '$bibcodes'}, 
                     'author_count': {'$size': '$author_set'}, 
                     'first_author_count': {'$size': '$first_author_set' }, 
                     '_id': 1, 
                     'count': 1}}
        sort = {'$sort': {'_id': 1}}
        pipeline = [ match, unwind, group, project, sort ]

        result = list(self.collection.aggregate(pipeline))
        return result

    def article_exists(self, article):
        """Check if an article exists in the collection."""
        return self.collection.count_documents({'$or': [{'id': article['id']}, {'bibcode': article['bibcode']}]}) > 0

    def get_articles(self, begin_year=None, end_year=None, month=None, affiliation=None):
        """Get articles from the collection."""
        query = {}
        if affiliation:
            query['affiliation'] = affiliation

        if not begin_year:
            begin_year = end_year

        if not end_year:
            query['year'] = begin_year
        else:
            query['year'] = {'$gte': begin_year, '$lte': end_year}

        if month:
            query['month'] = month

        rows = list(self.collection.find(
            query).sort('date', pymongo.DESCENDING))
        return rows

    def select_for_spreadsheet(self):
        """Select documents for spreadsheet export."""
        query = {'mission': {'$ne': 'unrelated'}}
        projection = {'bibcode': 1, 'year': 1, 'month': 1, 'date': 1, 'mission': 1,
                      'metrics': 1, 'affiliation': 1, 'date_modified': 1, 'last_modifier': 1, '_id': 0}
        rows = list(self.collection.find(query, projection).sort(
            'bibcode', pymongo.ASCENDING))
        return rows

    def get_articles_by_mission_years(self, mission, year_begin, year_end):
        """Get articles by mission and year range."""
        query = {
            'mission': mission,
            'affiliation': 'keck',
            'year': {'$gte': year_begin, '$lte': year_end}
        }
        projection = {'year': 1, 'metrics': 1, '_id': 0}
        rows = list(self.collection.find(query, projection))
        return rows

    def get_articles_by_years_instrument(self, year_begin, year_end, instrument=None):
        """Get articles by year range, and instrument."""
        pipeline = []
        query = { 'year': {'$gte': year_begin, '$lte': year_end }, 'affiliation': 'keck' }
        group = {'_id': {'year': '$year'}, 'count': {'$sum': 1}}
        if instrument:
            pipeline.append({'$unwind': '$instruments'})
            query['instruments'] = instrument
            group['_id']['instrument'] = '$instruments'
        pipeline.append({'$match': query})

        sort = {'$sort': {'year': 1}}
        pipeline.append({'$group': group})
        pipeline.append(sort)
        rows = list(self.collection.aggregate(pipeline))
        # build a dict with all years in the range. 
        yeardict = {year: 0 for year in range(year_begin, year_end + 1)}
        for row in rows:
            yeardict[row['_id']['year']] = row['count']
        return yeardict

    def get_count_cumulative(self, year):
        """Get cumulative count of articles by mission and year."""
        query = {
            'year': {'$lte': year, 'affiliation': 'keck'},
        }
        count = self.collection.count_documents(query)
        return count
