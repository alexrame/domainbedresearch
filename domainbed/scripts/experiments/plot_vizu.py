dict_key_to_label = {"df": "Div. feats", "dr": "Div. preds.", "hess": "Flatness", "soup": "Acc.", "net": "Ens Acc."}
key1 = "hess"
key2 = "soup"
plt.get_xlabel(dict_key_to_label[key1])
plt.ylabel(dict_key_to_label[key2])


def plot_with_int(l, color, label):
    m, b = np.polyfit(get_x(l, key1), get_x(l, key2), 1)
    plt.plot(get_x(l, key1), m * np.array(get_x(l, key1)) + b, color=color)# label="int."+label)
    plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)

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
plt.get_xlabel(dict_key_to_label[key1])
plt.ylabel(dict_key_to_label[key2])
plt.scatter(get_x(ls, key1), get_x(ls, key2), color="red", label="swa")

m, b = np.polyfit(get_x(ls, key1), get_x(ls, key2), 1)
plt.plot(
    get_x(ls, key1),
    m * np.array(get_x(ls, key1)) + b,
    label="linear int."
)

plt.legend()


def plot_key(key1, key2, order=1):

    plt.xlabel(dict_key_to_label.get(key1, key1))
    plt.ylabel(dict_key_to_label.get(key2, key2))

    def plot_with_int(l, color, label):
        t = get_x(l, key1)
        if t == []:
            return
        if order == 1:
            m, b = np.polyfit(get_x(l, key1), get_x(l, key2), 1)
            plt.plot(get_x(l, key1), m * np.array(get_x(l, key1)) + b, color=color, label=label +": " + "{:.0f}".format(m*1000))
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color)
        elif order == 2:
            m2, m1, b = np.polyfit(get_x(l, key1), get_x(l, key2), 2)
            get_x1_sorted = sorted(get_x(l, key1))
            preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(get_x1_sorted, preds, color=color)# label="int."+label)
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)
        elif order == 3:
            m3, m2, m1, b = np.polyfit(get_x(l, key1), get_x(l, key2), 3)
            get_x1_sorted = sorted(get_x(l, key1))
            preds = m3 * np.array(get_x1_sorted)**3 + m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(get_x1_sorted, preds, color=color)# label="int."+label)
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)
        elif order == "2log":
            m2, m1, b = np.polyfit(np.log(get_x(l, key1)), get_x(l, key2), 2)
            get_x1_sorted = np.log(sorted(get_x(l, key1)))
            preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(sorted(get_x(l, key1)), preds, color=color)# label="int."+label)
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)

    colors = cm.rainbow(np.linspace(0, 1, 12))
    for card in range(2, 10):
        #print(card, l[card])
        plot_with_int(l[card], color=colors[card], label="swa" + str(card))

    plt.legend()
