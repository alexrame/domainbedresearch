dict_key_to_label = {"df": "Div. feats", "dr": "Div. preds.", "hess": "Flatness", "soup": "Acc.", "net": "Ens Acc."}
key1 = "hess"
key2 = "soup"
plt.xlabel(dict_key_to_label[key1])
plt.ylabel(dict_key_to_label[key2])


def plot_with_int(l, color, label):
    m, b = np.polyfit(x(l, key1), x(l, key2), 1)
    plt.plot(x(l, key1), m * np.array(x(l, key1)) + b, color=color)# label="int."+label)
    plt.scatter(x(l, key1), x(l, key2), color=color, label=label)

plot_with_int(l55, color="blue", label="soup55")
plot_with_int(l53, color="yellow", label="soup53")
plot_with_int(l33, color="pink", label="soup33")
plot_with_int(ls, color="red", label="swa")

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
