dict_key_to_label = {"df": "Div. feats", "dr": "Div. preds.", "hess": "Flatness", "soup": "Acc.", "net": "Ens Acc."}
key1 = "hess"
key2 = "net"
plt.xlabel(dict_key_to_label[key1])
plt.ylabel(dict_key_to_label[key2])
plt.scatter(x(l55, key1), x(l55, key2), color="blue", label="soup55")
plt.scatter(x(l53, key1), x(l53, key2), color="yellow", label="soup53")
plt.scatter(x(l33, key1), x(l33, key2), color="pink", label="soup33")
plt.scatter(x(ls, key1), x(ls, key2), color="red", label="swa")

m, b = np.polyfit(x(ls + l55 + l53 + l33, key1), x(ls + l55 + l53 + l33, key2), 1)
plt.plot(
    x(ls + l55 + l53 + l33, key1),
    m * np.array(x(ls + l55 + l53 + l33, key1)) + b,
    label="linear int."
)

plt.legend()



dict_key_to_label = {
    "df": "Div. feats",
    "dr": "Div. preds.",
    "hess": "Flatness",
    "soup": "Acc.",
    "net": "Ens Acc."
}
key1 = "div"
key2 = "soup"
plt.xlabel(dict_key_to_label[key1])
plt.ylabel(dict_key_to_label[key2])
plt.scatter(x(ls, key1), x(ls, key2), color="red", label="swa")

m, b = np.polyfit(x(ls, key1), x(ls, key2), 1)
plt.plot(
    x(ls, key1),
    m * np.array(x(ls, key1)) + b,
    label="linear int."
)

plt.legend()
