from re import sub
import collections
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_noun_chunks")

text = "Cognitive science is the study of the human mind and brain, focusing on how the mind represents and manipulates knowledge and how mental representations and processes are realized in the brain."

doc = nlp(text)

def camel_case(s):
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return ''.join([s[0].lower(), s[1:]])

def dictToFactPl(knowledgeBaseFile, dic):
    file = open(knowledgeBaseFile, "a")
    for pred, arg in dic.items():
        print(pred)
        predCamelCase = camel_case(pred)
        fact = predCamelCase + f"{arg}" + "." + "\n"
        file.writelines(fact)

        return fact
    file.close()

def mergeTokensUponSideRelativity(tokenCompound: list, tokenComplement: list) -> list:
    mergedTokenList = tokenCompound + tokenComplement
    mergedTokenIndexDict = {}

    for tok in mergedTokenList:
        mergedTokenIndexDict[tok.i] = tok

    orderdDict = collections.OrderedDict(sorted(mergedTokenIndexDict.items()))

    return list(orderdDict.values())

def demlimiter(seq: list) -> dict:
    # function that creates dict using delimiter as key and its set as value
    result_dict = {}
    l = []
    for token in seq:
        # the hypothesis is that object sequence ends with object itself
        if token.dep_ in ["pobj", "dobj", "obj", "conj"]:
            # compile obj and its complementation then add to dict, obj as key
            l.append(token)
            result_dict[token.text] = ' '.join([token.text for token in l])
            # empty the list
            l = []
        else:
            # fill the list with object complementation
            l.append(token)
    return result_dict

def getVerbConj(verb) -> tuple:
    # Condition - verb has conjuctions
    allConjCompl = []
    allConjs = []
    for child in verb.children:
        if child.dep_ == "conj":
            allConjs.append(child)
            conjCompl = getVerbComplementation(child)
            conjComplStr = [token.text for token in conjCompl]

            allConjCompl.append(' '.join(conjComplStr))

            c, cc = getVerbConj(child) # recursion
            for conj in cc:
                allConjCompl.append(conj)
            for co in c:
                allConjs.append(co)

    return allConjs, allConjCompl

def getVerbComplementation(verb) -> list:
    verbComplement = [verb]
    for verbChild in verb.children:
        if verbChild.dep_ in ("neg", "aux", "auxpass", "advmod", "attr", "prt"):
            verbComplement = mergeTokensUponSideRelativity([verbChild], verbComplement)
        elif verbChild.dep_ == "acomp" and verbChild.pos_ in ("VERB", "AUX", "ADJ", "NOUN"):
            verbAcomp = getVerbComplementation(verbChild) # recursion
            verbComplement = mergeTokensUponSideRelativity(verbAcomp, verbComplement)
        elif verbChild.dep_ == "xcomp" and verbChild.pos_ in ("VERB", "AUX", "ADJ", "NOUN"):
            verbXcomp = getVerbComplementation(verbChild) # recursion
            verbComplement = mergeTokensUponSideRelativity(verbXcomp, verbComplement)

    return verbComplement

def getVerbObject(verb) -> list:

    verbObjects = []
    for verbChild in verb.children:
        # if verbs' direct child is object
        # TODO: conj should have the same complementation as obj?
        if verbChild.dep_ in ("pobj", "dobj", "obj", "conj") and verbChild.pos_ != "VERB": # because verb can't be an object
            objectComp = getVerbObject(verbChild) # recursion
            objectCompWithPrep = objectComp + [verbChild]
            verbObjects = mergeTokensUponSideRelativity(objectCompWithPrep, verbObjects)

        elif verbChild.dep_ in ("prep", "pcomp", "agent", "acl"): ## acl ????
            objectComp = getVerbObject(verbChild) # recursion
            objectCompWithPrep = objectComp + [verbChild]
            verbObjects = mergeTokensUponSideRelativity(objectCompWithPrep, verbObjects)

        elif verb.pos_ == "AUX" and verbChild.dep_ == "attr":
            objectComp = getVerbObject(verbChild)  # recursion
            verbObjects = mergeTokensUponSideRelativity(objectComp, verbObjects)

    return verbObjects

def getSPO(subject) -> dict:
    spo = {}

    subjectList = [token.lemma_.lower() for token in list(subject.subtree)]
    subjectComplement = ' '.join(subjectList)

    verbComplementList = [token.text for token in getVerbComplementation(subject.head) if subject.head.pos_ in ["VERB", "AUX"]]
    predicate = ' '.join(verbComplementList)

    res = demlimiter(getVerbObject(subject.head))
    objects = list(res.values())
    spo[predicate] = (subjectComplement, objects)

    conj, conjCompl = getVerbConj(subject.head)
    conjDic = dict(zip(conj, conjCompl))

    for c, cpredicate in conjDic.items():
        spo[cpredicate] = (subjectComplement, getVerbObject(c))

    return spo

def getSPODependecy(text) -> list:
    SPOlist = []
    # TODO: first conduct coreference resolution on whole doc
    for sent in text.sents:
        for token in sent:
            # token is subject
            if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
                SPOlist.append(getSPO(token))

    return SPOlist

print("List of facts:", getSPODependecy(doc))
print(f"Number of facts: {len(getSPODependecy(doc))}")

