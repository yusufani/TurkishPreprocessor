# Yazar : Yusuf ANI
import re
from turkish.deasciifier import Deasciifier
import multiprocessing
import threading
from tqdm import tqdm
import nltk
import logging


# Source : https://towardsdatascience.com/text-normalization-7ecc8e084e31
class Preprocessor:
    def __init__(self, processors="all", deleted_processors=False, additional_processors=None, dont_print=False):
        log = logging.getLogger()
        if dont_print:
            log.setLevel(logging.ERROR)
        else:
            log.setLevel(logging.INFO)

        main_processors = ["delete_hyperlinks", "simplify_punctuation", "normalize_whitespace", "correct_letters",
                           "clear_stop_words"]
        extra_processors = ["lower_text", "delete_short_texts", "replace_ampersand",
                            "delete_ordered_hashtag_usernames", "delete_numeric_values", "lemmatization"]
        self.stop_words = nltk.corpus.stopwords.words('turkish')

        if type(processors) == str and processors.lower() == "all":
            self.selected_processors = main_processors
        elif deleted_processors and len(processors) > 1 and type(processors) == list:
            # This means that I will delete processors variable processors from all processors
            self.selected_processors = [i for i in main_processors if i not in processors]
        elif len(processors) > 1 and type(processors) == list:
            self.selected_processors = processors
        else:
            raise Exception(" There is error in processors selecting. Check library code for more info")

        if type(additional_processors) == str and additional_processors.lower() == "all":
            self.selected_processors += extra_processors
        elif type(additional_processors) == list and len(additional_processors) >= 1:
            self.selected_processors += additional_processors
        elif additional_processors is None:
            print("Extra processors not selected")
        else:
            raise Exception(" There is error in additional processors selecting. Check library code for more info")

        print(50 * "*")
        print("All Selected processors are : ")
        for i in self.selected_processors:
            print(i)
        print(50 * "*")

    def preprocess_texts(self, text_list):
        '''
        List of texts for processing 
        '''
        n_threads = multiprocessing.cpu_count()
        logging.info("Number of threads will executed: "+ str(n_threads))
        self.text_list = text_list
        threads = []
        for idx, val in tqdm(enumerate(text_list)):
            threads.append(threading.Thread(target=self.preprocess_text, args=(val, idx,)))
            if (idx + 1) % n_threads == 0 or idx == len(text_list) - 1:
                for i in threads: i.start()
                for i in threads: i.join()
                threads = []

        return text_list

    def preprocess_text(self, text, idx=-1):

        if type(text) == float and str(text) == "nan":
            if idx != -1:  # Multi thread
                self.text_list[idx] = None
                return
            else:
                return None

        if "lower_text" in self.selected_processors:
            logging.info("Text is lowering...")
            text = self.lower_text(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nText is lowered")
        if "delete_hyperlinks" in self.selected_processors:
            logging.info("Hyperlinks are deleting...")
            text = self.delete_hyperlinks(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nHyperlinks are deleted")
        if "delete_numeric_values" in self.selected_processors:
            logging.info("Numeric values are deleting...")
            text = self.delete_numeric_values(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nNumeric values are deleted")
        if "simplify_punctuation" in self.selected_processors:
            logging.info("Punctuation is correcting...")
            text = self.simplify_punctuation(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nPunctuation is corrected")
        if "normalize_whitespace" in self.selected_processors:
            logging.info("Whitespace is normalizing...")
            text = self.normalize_whitespace(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nWhitespace is normalized")
        if "clear_stop_words" in self.selected_processors:
            logging.info("Stop Words is clearing...")
            text = self.clear_stop_words(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nStop Words is cleared...")
        if "delete_ordered_hashtag_usernames" in self.selected_processors:
            logging.info("Deleting Ordered hashtag&usernames...")
            text = self.delete_ordered_hashtag_usernames(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nDeleted Ordered hashtag&usernames...")
        if "replace_ampersand" in self.selected_processors:
            logging.info("Ampersand fixing...")
            text = self.replace_ampersand(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nAmpersand fixed...")
        if "correct_letters" in self.selected_processors:
            logging.info("Letters correcting...")
            text = self.correct_letters(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nLetters corrected...")
        if "delete_short_texts" in self.selected_processors:
            logging.info("Deleting short texts...")
            text = self.delete_short_texts(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nDeleted short texts...")
        if "lemmatization" in self.selected_processors:
            logging.info("words Lemmatizing...")
            text = self.lemmatization(text)
            logging.info("Input Text: " + text + "\n Output Text:" + text + "\nWords lemmatized")

        if idx != -1:  # Multi thread
            self.text_list[idx] = text.strip()
            return
        else:
            return text.strip()

    @staticmethod
    def lemmatization(text):
        from .Turkish_Lemmatizer.lemmatizer import get_lem
        return get_lem(text)

    @staticmethod
    def delete_hyperlinks(text):

        try:
            return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text).strip()
        except Exception as e:
            print("t:", text)
            print(type(text))

    @staticmethod
    def simplify_punctuation(text):
        """
        This function simplifies doubled or more complex punctuation. The exception is '...'.
        """
        corrected = str(text)
        corrected = re.sub(r'([!?,;:])\1+', r'\1', corrected)
        corrected = re.sub(r'\.{2,}', r'...', corrected)
        corrected = re.sub(r'\s([?.!:"](?:\s|$))', r'\1', corrected) # Delete space before punctuation

        return corrected

    @staticmethod
    def normalize_whitespace(text):
        """
        This function normalizes whitespaces, removing duplicates.
        """
        corrected = str(text)
        corrected = re.sub(r"//t", r"\t", corrected)
        corrected = re.sub(r"( )\1+", r"\1", corrected)
        corrected = re.sub(r"(\n)\1+", r"\1", corrected)
        corrected = re.sub(r"(\r)\1+", r"\1", corrected)
        corrected = re.sub(r"(\t)\1+", r"\1", corrected)
        return corrected.strip(" ")

    @staticmethod
    def normalize_contractions(text):
        raise Exception("Not Implemented")
        # TODO
        pass

    @staticmethod
    def lower_text(text):
        return text.lower()

    @staticmethod
    def delete_numeric_values(text):
        return ''.join(filter(lambda x: not x.isdigit(), text))

    def clear_stop_words(self, text):
        words = [word for word in text.split() if word not in self.stop_words]
        return " ".join(words)

    @staticmethod
    def delete_ordered_hashtag_usernames(text):
        '''
        This function delete ordered hashtag or username
        ! It will not delete single hashtag
        For example :
        bla bla #bla #bla @bla #bla -> bla bla
        bla @bla bla -> bla bla bla
        bla #bla bla ->

        :param text: string
        :return: text : string
        '''
        fields = text.split(" ")
        text = []
        temp_text = []
        for field in fields:
            if "@" in field or "#" in field:
                temp_text.append(field)
            else:
                if len(temp_text) == 1:
                    text.append(temp_text[0].replace("@", "").replace("#", ""))
                    temp_text = []
                text.append(field)

        if len(text) >= 1:
            return " ".join(text).strip()
        else:
            return ""

    @staticmethod
    def replace_ampersand(text):
        return text.replace("&amp;", "&")

    @staticmethod
    def delete_short_texts(text, n=1):
        '''

        :param text:
        :param n: Threshold for min deleting word
        :return:
        '''
        # Eğer 2 kelime ise sil
        if len(text.split(" ")) <= n:
            # print(text)
            return ""
        return text

    @staticmethod
    def correct_letters(text):
        return Deasciifier(text).convert_to_turkish()
# %%
